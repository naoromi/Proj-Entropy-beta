import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        # padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride=stride, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size[1], stride=stride, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)      # for completionist sanity sake
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size[2], stride=stride, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # expand in_channels to equal out_channel to enable element-wise addition
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        else:
            self.shortcut = nn.Sequential()
        
        self.bn4 = nn.BatchNorm1d(out_channels)     # exclusively for shortcut, to bn raw inputs
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)) + self.bn4(self.shortcut(x)))
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.layer1 = ResidualBlock(input_shape[0], n_feature_maps, kernel_size=[8, 5, 3])
        self.layer2 = ResidualBlock(n_feature_maps, n_feature_maps * 2, kernel_size=[8, 5, 3])
        self.layer3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2, kernel_size=[8, 5, 3])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        # out = out.view(out.size(0), -1)     # imo, ambiguous intent
        out = torch.flatten(out, 1) # flatten all dimensions except batch
        out = self.fc(out)
        return out


