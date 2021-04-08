import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
from maml.data.get_loader import get_visda
from maml.base.model import ResNet18

batch_size = 32
n_class_train = 200
n_class = 10
n_shots = 10

f = open("./saved/classes.txt", "r")
train_classes = []
lines = f.readlines()
f.close()
for line in lines:
    train_classes.append(int(line))

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
domain_acc = {domain: 0 for domain in domains}


n_mc = 100

for i, mc in enumerate(range(n_mc)):
    possible_classes = [c for c in range(345) if c not in train_classes]
    n_poss = len(possible_classes)
    indexes = list(np.random.choice(range(n_poss), n_class, replace=False))
    classes = [possible_classes[i] for i in indexes]

    trainloader, testloader = get_visda(
        batch_size, 8, "/home/louishemadou/VisDA", ["real"], 0.7, classes, n_shots)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = ResNet18()
    encoder.to(device)
    encoder.eval()

    classifier = torch.load("./saved/classifier_visda")
    num_features = classifier.layers[9].in_features

    classifier.layers[9] = nn.Linear(num_features, n_class)
    classifier.to(device)

    n_epoch = 20
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    acc_list_train = []
    acc_list_test = []

    T0 = time.time()

    for epoch in range(n_epoch):

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = encoder(inputs)

            classifier.train()

            optimizer.zero_grad()
            out = classifier(features)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            message = "MC {}/{} Epoch {}/{} ({:.0f}%) \r".format(mc+1, n_mc, epoch+1,
                                                              n_epoch, 100*i/len(trainloader))
            print(message, sep=" ", end="", flush=True)

    classifier.eval()

    def accuracy(loader):
        correct = 0
        with torch.no_grad():
            for val_imgs, val_labels in loader:
                val_imgs = val_imgs.to(device).float()
                val_labels = val_labels.to(device)
                out = classifier(encoder(val_imgs))
                predicted = torch.max(out, 1)[1].to(device)
                correct += (predicted == val_labels).sum()
            correct_percent = 100*correct/(len(loader)*batch_size)
        return correct_percent

    for domain in domains:

        _, target_test = get_visda(
            batch_size, 8, "/home/louishemadou/VisDA", [domain], 0.5, classes)

        domain_acc[domain] += accuracy(target_test)/n_mc
print(domain_acc)
