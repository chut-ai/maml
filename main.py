from maml.data.dataset import VisdaDataset, get_index_to_class
from maml.data.get_loader import get_visda
from maml.model import ConvNet
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64

trainloader, testloader = get_visda(batch_size, 8, "/home/louishemadou/VisDA", "real", 0.8, classes = [1, 2, 3, 4, 5, 6, 7])

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

n_epoch = 50

for epoch in range(n_epoch):

    correct = 0
    
    for i, data in enumerate(trainloader, 0):


        inputs, labels = data

        optimizer.zero_grad()

        out = net(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        predicted = torch.max(out.data, 1)[1]
        correct += (predicted == labels).sum()
        correct_percent = correct*100/(batch_size*(i+1))

        if i%10 == 0:
            message = "Epochs : {}/{}, ({:.0f}%), Loss:{:.6f}, Accuracy:{:.3f}".format(epoch+1, n_epoch, 100*i/len(trainloader), loss, correct_percent)
            print(message)

