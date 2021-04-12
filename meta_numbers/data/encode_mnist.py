import os
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms, datasets


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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Resize(224)
    ])
MNIST = datasets.MNIST("~/MNIST/train", download=False, train=True, transform=transform)

MNIST_loader = torch.utils.data.DataLoader(MNIST, 1, shuffle=True)
encoded_dict = {y:[] for y in range(10)}
encoder = ResNet18().cuda()
encoder.eval()

n_img = 5000

for i, (x, y) in enumerate(MNIST_loader):
    message = "Encoding MNIST images, ({:.1f}%)".format(100*i/n_img)
    print(message, end="\r", flush=True)
    with torch.no_grad():
        x = x.cuda()
        x = encoder(x).squeeze().cpu().tolist()
    encoded_dict[int(y.data)].append(x)
    if i > n_img:
        break
    
with open("/home/louishemadou/MNIST_encoded/mnist.json", "w") as outfile:
    json.dump(encoded_dict, outfile)

