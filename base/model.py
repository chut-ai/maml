import torch
import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        resnet18 = models.resnet18(pretrained=True)
        self.up = nn.Upsample((224, 224))
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool

        self.in_features = resnet18.fc.in_features

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class LastLayers(nn.Module):
    def __init__(self, num_features, n_class):
        super(LastLayers, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_class))

    def forward(self, x):
        x = self.layers(x)
        return x
