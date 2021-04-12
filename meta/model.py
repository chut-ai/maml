import torch.nn as nn


class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
            )
    def forward(self, x):
        x = self.classifier(x)
        return x


class SmallDenseNet(nn.Module):

    def __init__(self):
        super(SmallDenseNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
            )
    def forward(self, x):
        x = self.classifier(x)
        return x
