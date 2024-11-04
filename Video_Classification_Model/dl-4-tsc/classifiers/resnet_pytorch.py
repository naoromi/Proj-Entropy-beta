import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib
import csv
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.utils import save_test_duration, save_logs, calculate_metrics


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.layer1 = ResidualBlock(input_shape[0], n_feature_maps, kernel_size=7)
        self.layer2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_size=5)
        self.layer3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Classifier_RESNET:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if build:
            self.model = ResNet(input_shape, nb_classes).to(self.device)
            if self.verbose:
                print(self.model)
            if load_weights:
                self.model.load_state_dict(torch.load(os.path.join(self.output_directory, 'model_init.pth')))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.output_directory, 'model_init.pth'))

    def fit(self, x_train, y_train, x_val, y_val, train_truth, val_truth):
        if not torch.cuda.is_available():
            print('Error: GPU not available.')
            exit()

        batch_size = 64
        nb_epochs = 150
        mini_batch_size = min(x_train.shape[0] // 10, batch_size)

        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        x_val = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)

        start_time = time.time()

        train_acc=[]
        test_acc=[]
        for epoch in range(nb_epochs):
            self.model.train()
            permutation = torch.randperm(x_train.size()[0])
            for i in range(0, x_train.size()[0], mini_batch_size):
                indices = permutation[i:i + mini_batch_size]
                batch_x, batch_y = x_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(x_val)
                val_loss = criterion(val_outputs, y_val)
                scheduler.step(val_loss)
                
                # predict
                train_pred = self.model(x_train).numpy()
                val_pred = self.model(x_val).numpy()
                
                # calculate metrics
                train_pred = np.argmax(train_pred, axis=1)
                val_pred = np.argmax(val_pred, axis=1)
                
                df_metrics = calculate_metrics(train_truth, train_pred,
                                               0.0,
                                               val_truth, val_pred)
                print(df_metrics)

                train_acc.append(df_metrics['accuracy'])
                test_acc.append(df_metrics['accuracy_val'])

            if self.verbose and epoch % 1 == 0:
                print(f'Epoch {epoch}/{nb_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')
        
        plt.figure()
        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.title('model_acc')
        plt.ylabel("acc", fontsize='large')
        plt.xlabel('epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("fig.png", bbox_inches='tight')
        plt.close()
        with open("output.csv", 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['train_acc']+[float(i) for i in train_acc])
            writer.writerow(['test_acc']+[float(i) for i in test_acc])
        return 0

    def predict(self, x_test, y_true, return_df_metrics=True):
        start_time = time.time()
        model_path = os.path.join(self.output_directory, 'last_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        x_test = x_test.to(self.device)
        with torch.no_grad():
            y_pred = self.model(x_test).cpu().numpy()

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(os.path.join(self.output_directory, 'test_duration.csv'), test_duration)
            return y_pred