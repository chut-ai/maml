import torch.nn as nn


class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
            )
    def forward(self, x):
        x = self.classifier(x)
        return x
