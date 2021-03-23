import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import numpy as np
import matplotlib.pyplot as plt
from maml.data.get_loader import get_visda

batch_size = 32
n_class = 200

classes = list(np.random.choice(range(334), n_class, replace=False))

trainloader, testloader = get_visda(
    batch_size, 8, "/home/louishemadou/VisDA", "real", 0.7, classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = models.resnet18(pretrained=True)

# Freeze all parameters of the model

for param in net.parameters():
    param.requires_grad = False

# Create the last fc, by default requires_grad=True
num_features = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, n_class))


net = net.to(device)
print(net)

n_epoch = 10
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

acc_list = []

T0 = time.time()

for epoch in range(n_epoch):

    for i, data in enumerate(trainloader, 0):

        net.train()

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        message = "Epochs: {}/{}, ({:.0f}%) \r".format(epoch+1,
                                                       n_epoch, 100*i/len(trainloader))
        print(message, sep=" ", end="", flush=True)

    net.eval()
    correct = 0
    for val_imgs, val_labels in testloader:
        val_imgs = val_imgs.to(device).float()
        val_labels = val_labels.to(device)
        out = net.forward(val_imgs)
        predicted = torch.max(out, 1)[1].to(device)
        correct += (predicted == val_labels).sum()
    correct_percent = 100*correct/(len(testloader)*batch_size)

    acc_list.append(correct_percent.cpu())
    print("\r")
    elapsed = (time.time()-T0)/60
    print("Test accuracy: {:.3f}, time elapsed (in minuts):{:.0f}".format(
        correct_percent, elapsed))

torch.save(net, "./resnet34_visda")


X = range(len(acc_list))

plt.figure(1)
plt.plot(X, acc_list)
plt.title("Accuracy over epochs")
plt.show()
