import os
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms


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

def get_label(img_name):
    label = int(img_name[-14: -11])
    return label


def encode(domain, json_file):
    
    encoder = ResNet18().cuda()
    encoder.eval()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, (1, 1), (1, 1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    encoded_dict = {label: [] for label in range(1, 346)}
    imgs_path = "/home/louishemadou/VisDA/" + domain + "/"

    img_list = os.listdir(imgs_path)

    for i, img in enumerate(img_list):
        message = "Encoding {} images, ({:.1f}%) \r".format(domain, 100*i/len(img_list))
        print(message, sep=" ", end="", flush=True)
        label = get_label(img)
        x = Image.open(imgs_path + img)
        x = transform(x)
        x = x.unsqueeze(0).cuda()
        with torch.no_grad():
            x = encoder(x).squeeze().cpu().tolist()
        encoded_dict[label].append(x)

    with open(json_file, "w") as outfile:
        json.dump(encoded_dict, outfile)

encode("real", "/home/louishemadou/VisDA_encoded/real.json")
encode("clipart", "/home/louishemadou/VisDA_encoded/clipart.json")
encode("infograph", "/home/louishemadou/VisDA_encoded/infograph.json")
encode("sketch", "/home/louishemadou/VisDA_encoded/sketch.json")
encode("painting", "/home/louishemadou/VisDA_encoded/painting.json")
encode("quickdraw", "/home/louishemadou/VisDA_encoded/quickdrawjson")
