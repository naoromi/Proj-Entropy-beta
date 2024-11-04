import os, time, math, csv
import pandas as pd
import numpy as np
from os.path import join
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.rcParams["font.family"] = "sans-serif"

import wandb
from torchinfo import summary
from .utils.fg_utils import preflight_version_check
from utils.utils import create_directory
from .model_resnet_tsc import ResNet


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path (crude synchronization)
            return None
        return directory_path

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float32), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def train_model(design_matrix_path, save_path, architecture, gpus):
    # for some reason, by checking cuda availability first, makes it available...
    preflight_version_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------
    #    Data Loading + Transform + Create Datasets & Dataloaders
    # ----------------------------------------------------------------------

    # Load data from the DataFrame
    X = pd.read_csv(join(design_matrix_path, "X_train.csv")).values
    y = pd.read_csv(join(design_matrix_path, "Y_train.csv")).values

    # permute and split train/val set (why train_test_split when I could just use numpy)
    percentage = 0.1
    threshold = math.floor(len(X)*percentage)
    np.random.seed(42)          # seed permutation for reproducibility
    shuffled_indices = np.random.permutation(len(X))
    train_idx, val_idx = shuffled_indices[threshold:], shuffled_indices[:threshold]

    # train_test_split in pure numpy
    x_train, y_train, x_val, y_val = (X[train_idx, :],
                                    y[train_idx, :],
                                    X[val_idx, :],
                                    y[val_idx, :])

    # one hot encode categorical variable
    enc = OneHotEncoder(categories='auto')
    enc.fit(y.reshape(-1, 1))            # just formality, y already in good shape
    y_train = enc.transform(y_train).toarray()
    y_val = enc.transform(y_val).toarray()

    nb_classes = len(enc.categories_[0])
    

    # normalize X (each example individually)
    x_train_mean, x_train_std = x_train.mean(axis=1, keepdims=True), x_train.std(axis=1, keepdims=True)
    x_train = (x_train - x_train_mean) / x_train_std

    x_val_mean, x_val_std = x_val.mean(axis=1, keepdims=True), x_val.std(axis=1, keepdims=True)
    x_val = (x_val - x_val_mean) / x_val_std


    # add an extra C dimension because model expects it
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1,x_val.shape[1]))


    ## creating data loaders
    # Convert the numpy array to a PyTorch tensor
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.float32))
    val_dataset   = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    ### Create a DataLoader from the TensorDataset
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)
    
    
    # ----------------------------------------------------------------------
    #    "Almost" Boilerplate Training
    # ----------------------------------------------------------------------
    
    # Hyperparameters
    nb_epochs = 1500
    early_stop_patience = 80  # Patience level for early stopping
    lr = 0.001
    min_lr = 0.00001
    lr_reduce_factor = 0.1
    lr_patience = 40
    
    # log configs to wandb, 
    # TODO: consider yacs style configs, downside: prepend 'config' everywhere
    wandb.config = {
        "nb_epochs": nb_epochs,
        "early_stop_patience": early_stop_patience,
        "learning_rate": lr,
        "lr_reduce_factor": lr_reduce_factor,
        "lr_patience": lr_patience
    },

    ### Create model
    input_shape = x_train.shape[1:]     # (C, N) = (1, 3000)
    model = ResNet(input_shape, nb_classes, 
                   n_feature_maps=architecture['num_feature_maps']).to(device)


    ### define: loss fn, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, 
                                patience=lr_patience, verbose=True)

    print('Begin Training...')
    ### loop
    train_acc, val_acc = [], []
    patience_cnt = 0

    for epoch in range(nb_epochs):
        start = time.perf_counter()
        
        ### training loop
        model.train()
        train_loss = 0
        train_logits, train_truth = [], []
        for batch_num, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            batch_loss = criterion(logits, y)
            batch_loss.backward()
            optimizer.step()
            
            # mark down results for metric calculations
            train_logits += logits
            train_truth += y
            train_loss += batch_loss
            
        train_loss /= len(train_dataloader.dataset)

        ### eval loop
        model.eval()
        with torch.no_grad():
            # eval
            val_loss = 0     # for lr scheduler
            val_logits, val_truth = [], []
            for batch_num, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_logits += logits
                val_truth += y
                val_loss += criterion(logits, y) * len(y)     # properly weighting each batch loss, 
            val_loss /= len(val_dataloader.dataset)
            
            # calculate metrics (bit wonky on logits, converting to numerical categories)
            
            metrics = calculate_metrics(
                torch.stack(train_truth).max(dim=1)[1].cpu(), 
                torch.stack(train_logits).max(dim=1)[1].cpu(), 
                0.0, 
                torch.stack(val_truth).max(dim=1)[1].cpu(), 
                torch.stack(val_logits).max(dim=1)[1].cpu())
            print(metrics)
            
            train_acc.append(metrics['accuracy'][0])
            val_acc.append(metrics['accuracy_val'][0])
            
            # step lr scheduler
            scheduler.step(val_loss)
            
            # print out metrics
            if epoch % 1 == 0 or epoch == nb_epochs-1:
                lr = scheduler.optimizer.param_groups[0]['lr']
                dt = time.perf_counter() - start
                
                print(f'Epoch {epoch+1}/{nb_epochs}' + \
                    f' | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}' + \
                    f' | Acc: {train_acc[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}' + \
                    f' | lr: {lr:.4f} | dt: {dt:.3f}s  ')
                
                wandb.log(
                    {
                        "train_loss": train_loss, 
                        "val_loss": val_loss,
                        "train_accuracy": train_acc[-1], 
                        "val_accuracy": val_acc[-1],
                        "lr": lr,
                        "runtime": dt
                    }
                )
                
                
        ### visualize on every round (poor man's tensorboard) ###

        plt.figure()
        plt.plot(train_acc)
        plt.plot(val_acc)

        plt.title('model_acc')
        plt.ylabel("acc", fontsize='large')
        plt.xlabel('epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(join(save_path, 'train_test_curve.png'), 
                    bbox_inches='tight')

        plt.close()

        with open(join(save_path, 'metrics.csv'),
                'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['train_acc']+[float(i) for i in train_acc])
            writer.writerow(['val_acc']+[float(i) for i in val_acc])
        
        ### end visualization ###
        
        
        ### Model Savings ###
        
        # always save the last model, in case of crash
        torch.save({
            'timestamp': datetime.today().strftime('%Y-%m-%d__%H-%M-%S'),
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc[-1],
            'val_acc': val_acc[-1],
        }, join(save_path, 'last_model.pt'))
        
        # and save best model by val acc
        if val_acc[-1] >= max(val_acc):
            patience_cnt = 0
            print(f'### Saving Best Model: val_acc = {val_acc[-1]} ###')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc[-1],
                'val_acc': val_acc[-1],
            }, join(save_path, 'best_model.pt'))
        else:       # Early stopping
            patience_cnt += 1
            if patience_cnt >= early_stop_patience:
                print("## Early stopping triggered. ##")
                break
        
        
    print('Training finished.')
    print(f"## Last Model saved to: {save_path} ##")

    
    

