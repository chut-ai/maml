import torch
import torch.nn as nn
from torchvision import models


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

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.upsample = nn.Upsample(224)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, 10)

    def forward(self, x):

        with torch.no_grad():
            x = self.upsample(x)
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
        x = self.fc(x)
        return x


