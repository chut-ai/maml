# class ResBlock(nn.Module):

    # """Conv3x3 -> BN -> ReLU -> Conv3x3 -> sum -> ReLU
       # ------------Conv1x1--------------->

       # (channel) -> (channel)
    # """

    # def __init__(self, in_channel, mid_channel, out_channel, pool):
        # super(ResBlock, self).__init__()

        # self.shortcut = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        # self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, padding=1)
        # self.bn = nn.BatchNorm2d(mid_channel)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(mid_channel, out_channel, 3, padding=1)
        # if pool:
            # self.pool = nn.MaxPool2d((2, 2))
        # else:
            # self.pool = nn.Identity()
        # self.relu2 = nn.ReLU(inplace=True)

    # def forward(self, x):
        # residual = self.shortcut(x)
        # x = self.conv1(x)
        # x = self.bn(x)
        # x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.pool(x + residual)
        # x = self.relu2(x)
        # return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(8*16*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)


        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 8*16*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=1)
