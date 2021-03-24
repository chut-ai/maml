import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import matplotlib.pyplot as plt
from maml.data.get_loader import get_visda

batch_size = 16
n_class_train = 200
n_class = 10

trainloader, testloader = get_visda(
    batch_size, 8, "/home/louishemadou/VisDA", "real", 0.7, range(334, 344))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get model trained with 200 classes
net = torch.load("./resnet34_visda")

# Freeze all parameters of the model
for param in net.parameters():
    param.requires_grad = True

# Create the last fc
num_features = net.fc[2].in_features
net.fc[2] = nn.Linear(num_features, n_class)

net = net.to(device)

n_epoch = 5
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

        message = "Epoch {}/{} ({:.0f}%) \r".format(epoch+1,
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
    elapsed = (time.time()-T0)/60
    print("\rEpoch {}, test accuracy: {:.3f}, {:.0f} minutes elapsed".format(epoch+1, correct_percent, elapsed))


net.eval()

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]

for domain in domains:

    _, target_test = get_visda(
        batch_size, 8, "/home/louishemadou/VisDA", "clipart", 0.7, range(334, 344))

    target_correct = 0
    for val_imgs, val_labels in target_test:
        val_imgs = val_imgs.to(device).float()
        val_labels = val_labels.to(device)
        out = net.forward(val_imgs)
        predicted = torch.max(out, 1)[1].to(device)
        target_correct += (predicted == val_labels).sum()
    target_correct_percent = 100*target_correct/(len(target_test)*batch_size)

    print("{} accuracy : {:.3f}".format(domain, target_correct_percent))
