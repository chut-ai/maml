import torch
import torch.nn as nn
from torchvision import models


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 345))

    def forward(self, x):
        x = self.layers(x)
        return x
