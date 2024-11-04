import os
from os.path import join

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from .utils.fg_utils import preflight_version_check
from utils.utils import save_test_duration, save_logs, calculate_metrics
from .model_resnet_tsc import ResNet
    
# and test the model
def load_and_test_model(design_matrix_path, model_path, gpus):
    # for some reason, by checking cuda availability first, make it available...
    preflight_version_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
   
    # ----------------------------------------------------------------------
    # Loading and Testing
    
    ### Data Loading + Transform + Create Datasets & Dataloaders ###

    # Load data from the DataFrame
    X = pd.read_csv("train-data.csv").values
    y = pd.read_csv("train-labels.csv").values
    
    # one hot encode categorical variable
    enc = OneHotEncoder(categories='auto')
    enc.fit(y.reshape(-1, 1))            # just formality, y already in good shape
    y = enc.transform(y).toarray()
    
    nb_classes = len(enc.categories_[0])
    
    
    # normalize X (each example individually)
    X_mean, X_std = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    X = (X - X_mean) / X_std
    
    # add an extra C dimension because model expects it
    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1,x_val.shape[1]))

    ## creating data loaders
    # Convert the numpy array to a PyTorch tensor
    test_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))

    ### Create a DataLoader from the TensorDataset
    batch_size = 64
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # ----------------------------------------------------------------------
    # Test Loop
    
    # Hyperparameters
    nb_epochs = 1000
    early_stop_patience = 80  # Patience level for early stopping
    lr = 0.001
    min_lr = 0.00001
    lr_reduce_factor = 0.1
    lr_patience = 40

    ### Create model
    input_shape = x_train.shape[1:]     # (C, N) = (1, 3000)
    model = ResNet(input_shape, nb_classes).to(device)

    ### define: loss fn, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, 
                                patience=lr_patience, verbose=True)

    checkpoint = torch.load(os.path.join(output_directory, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

    model.eval()

    test_loss = 0
    test_logits, test_truth = [], []
    with torch.no_grad():
        for batch_num, (X, y) in enumerate(test_dataloader):
            logits = model(X)
            test_logits += logits
            test_truth += y
            test_loss += criterion(logits, y) * len(y) 
        test_loss /= len(test_dataloader.dataset)

    metrics = calculate_metrics(torch.stack(test_truth).max(dim=1)[1].cpu(), 
                                torch.stack(test_logits).max(dim=1)[1].cpu(), 
                                0.0)
    print(metrics)

