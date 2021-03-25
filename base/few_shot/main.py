import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import numpy as np
import matplotlib.pyplot as plt
from maml.data.get_loader import get_visda
from maml.base.few_shot.model import ResNet18, LastLayers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(encoder, classifier, loader, batch_size):
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

def train_and_eval(n_instances, n_monte_carlo):
    batch_size = n_instances
    classes = range(10)
    n_class = len(classes)

    domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
    accuracies = {domain: 0 for domain in domains}

    for n in range(n_monte_carlo):
        print(n_instances, n)
        trainloader = get_visda(batch_size, 8, "/home/louishemadou/VisDA", "real", 1, classes, n_instances=n_instances)

        encoder = ResNet18()
        encoder.to(device)
        encoder.eval()
        num_features = encoder.in_features
        classifier = LastLayers(num_features, n_class)
        classifier.to(device)

        n_epoch = 10
        optimizer = optim.Adam(classifier.parameters(), lr=0.005, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()

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

        classifier.eval()


        for domain in domains:

            _, testloader = get_visda(batch_size, 8, "/home/louishemadou/VisDA", domain, 0.7, classes)

            acc = accuracy(encoder, classifier, testloader, batch_size)
            
            accuracies[domain] = (1/(n+1))*(n*accuracies[domain] + acc)

    return accuracies

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
all_accuracies = {domain: [] for domain in domains}

n_monte_carlo = 20
n_instances = [2, 5, 10, 20]

for n_instance in n_instances:
    accuracies = train_and_eval(n_instance, n_monte_carlo)
    for domain, acc in accuracies.items():
        all_accuracies[domain].append(acc.item())

colors = ["k", "g", "r", "b", "y"]
plt.figure()
for (domain, values), color in zip(all_accuracies.items(), colors):
    plt.plot(n_instances, values, color, label=domain)
plt.legend()
plt.xlabel("Nombre de shots")
plt.ylabel("Précision")
plt.show()
