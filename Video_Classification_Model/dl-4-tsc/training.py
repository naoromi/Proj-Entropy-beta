import os, math, csv, time
import numpy as np
import pandas as pd
import sklearn
from utils.utils import create_directory
from classifiers.resnet_pth_fixed import ResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.rcParams["font.family"] = "sans-serif"
from utils.utils import save_test_duration, save_logs, calculate_metrics


# read in arguments



# setup save locations
classifier_name = 'resnet_pytorch'
archive_name = 'custom'
dataset_name = 'h264_all'
itr_num = 1

root_dir = "/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/dl-4-tsc"

print('Method: ', archive_name, dataset_name, classifier_name, itr_num)

output_directory_tmpl = f"{root_dir}/results/{classifier_name}/{archive_name}/{dataset_name}_itr_"
for i in range(1, 100):     # at most 100 iterations
    output_directory = create_directory(output_directory_tmpl+str(i))
    if output_directory != None:
        break
print(output_directory)


# loading in data
df_x = pd.read_csv("/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/3_model_input_data/resnet/penguin_archive/hevc_4400/X_train.csv").values[1:,:]
df_y = pd.read_csv("/mnt/hbnas/home/fgan/Proj-Entropy/Video_Classification_Model/3_model_input_data/resnet/penguin_archive/hevc_4400/Y_train.csv").values[1:]

# permute and split train/val set
percentage = 0.1
threshold = math.floor(len(df_x)*percentage)
shuffled_indices = np.random.permutation(len(df_x))
train_indicies, val_indicies = shuffled_indices[threshold:], shuffled_indices[:threshold]

x_train=df_x[train_indicies, :]
x_val=df_x[val_indicies, :]

# normalize each time series individually
x_train_mean, x_train_std = x_train.mean(axis=1, keepdims=True), x_train.std(axis=1, keepdims=True)
x_train = (x_train - x_train_mean) / x_train_std

x_val_mean, x_val_std = x_val.mean(axis=1, keepdims=True), x_val.std(axis=1, keepdims=True)
x_val = (x_val - x_val_mean) / x_val_std


if len(x_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension 
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_val = x_val.reshape((x_val.shape[0],  1,x_val.shape[1]))


# transform Y matrix
y_train = df_y[train_indicies]
y_val = df_y[val_indicies]
nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(df_y.reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()


# prepare argmax truth for later reference
train_truth = np.argmax(y_train, axis=1)
val_truth = np.argmax(y_val, axis=1)

# ----------------------------------------------------------------------
# Boilerplate Training
nb_epochs = 1500
batch_size = 128  # You can adjust the batch size as needed

## creating data loaders
# Convert the numpy array to a PyTorch tensor
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))

### Create a DataLoader from the TensorDataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)


### Create model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_shape = x_train.shape[1:]     # (C, N) = (1, 3000)
model = ResNet(input_shape, nb_classes).to(device)

### define: loss fn, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                                 patience=50, min_lr=0.0001)

### loop
train_acc=[]
test_acc=[]
for epoch in range(nb_epochs):
    start = time.perf_counter()
    
    ### training loop
    model.train()
    train_pred = []
    train_loss = 0
    for batch_num, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        batch_loss = criterion(logits, y)
        batch_loss.backward()
        optimizer.step()
        
        # get training metrics on the go
        max_vals, max_args = torch.max(logits, 1)     # reduce on dimension 1
        train_pred += list(max_args.cpu().numpy())
        train_loss += batch_loss
    train_loss /= len(train_dataloader.dataset)

    ### eval loop
    model.eval()
    with torch.no_grad():
        # eval
        val_pred = []
        val_loss = 0     # for lr scheduler
        for batch_num, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            max_vals, max_args = torch.max(logits, 1)     # reduce on dimension 1
            val_pred += list(max_args.cpu().numpy())
            val_loss += criterion(logits, y) * len(y)     # properly weighting each batch loss, 
        val_loss /= len(val_dataloader.dataset)
            
        # calculate metrics
        metrics = calculate_metrics(train_truth, train_pred, 0.0, val_truth, val_pred)
        print(metrics)
        
        train_acc.append(metrics['accuracy'][0])
        test_acc.append(metrics['accuracy_val'][0])
        
        # step lr scheduler
        scheduler.step(val_loss)
        
        # print out metrics
        if epoch % 1 == 0 or epoch == nb_epochs-1:
            lr = scheduler.optimizer.param_groups[0]['lr']
            dt = time.perf_counter() - start
            
            print(f'Epoch {epoch+1}/{nb_epochs}' + \
                  f' | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}' + \
                  f' | Acc: {train_acc[-1]:.4f} | Val Acc: {test_acc[-1]:.4f}' + \
                  f' | lr: {lr:.4f} | dt: {dt:.3f}s  ')
    
    # always save the last model, in case of crash
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc[-1],
        'val_acc': test_acc[-1],
    }, os.path.join(output_directory, 'last_model.pt'))
    
    # and save best model by val acc
    if test_acc[-1] >= max(test_acc):
        print(f'### Saving Best Model: val_acc = {test_acc[-1]} ###')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc[-1],
            'val_acc': test_acc[-1],
        }, os.path.join(output_directory, 'best_model.pt'))
        
        
    ### visualize on every round (mimic tensorboard)

    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)

    plt.title('model_acc')
    plt.ylabel("acc", fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(output_directory, 'train_test_curve.png'), 
                bbox_inches='tight')

    plt.close()

    with open(os.path.join(output_directory, 'metrics.csv'),
            'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['train_acc']+[float(i) for i in train_acc])
        writer.writerow(['test_acc']+[float(i) for i in test_acc])
    
    ### end visualization



# ----------------------------------------------------------------------
# Visualization

# currently no additional visualizations


# ----------------------------------------------------------------------
# Loading and Testing

checkpoint = torch.load(os.path.join(output_directory, 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']

model.eval()

val_pred = []
val_loss = 0
with torch.no_grad():
    for batch_num, (X, y) in enumerate(val_dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        max_vals, max_args = torch.max(logits, 1)     # reduce on dimension 1
        val_pred += list(max_args.cpu().numpy())
        val_loss += criterion(logits, y) * len(y)     # properly weighting each batch loss, 
    val_loss /= len(val_dataloader.dataset)

print(calculate_metrics([0], [0], 0.0, val_truth, val_pred))


