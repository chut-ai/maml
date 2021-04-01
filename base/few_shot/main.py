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


def compute_accuracy(encoder, classifier, loader, batch_size):
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


def single_train(trainloader, classes, batch_size):

    domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
    n_class = len(classes)

    encoder = ResNet18()
    encoder.to(device)
    encoder.eval()
    num_features = encoder.in_features
    classifier = LastLayers(num_features, n_class)
    classifier.to(device)

    n_epoch = 10
    optimizer = optim.Adam(classifier.parameters(),
                           lr=0.005, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):

        for data in trainloader:

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
    single_accuracy = {}

    for domain in domains:
        _, testloader = get_visda(
            batch_size, 8, "/home/louishemadou/VisDA", domain, 0.5, classes)
        acc = compute_accuracy(encoder, classifier, testloader, batch_size)
        single_accuracy[domain] = acc

    return single_accuracy


def train_and_eval(n_instance, n_mc_class, n_mc_instance):

    n_class = 10
    batch_size = n_instance

    domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
    accuracy = {domain: 0 for domain in domains}

    for i in range(n_mc_class):
        classes = list(np.random.choice(range(345), n_class, replace=False))

        for j in range(n_mc_instance):

            trainloader = get_visda(
                batch_size, 8, "/home/louishemadou/VisDA", "real", 1, classes, n_instance)

            single_accuracy = single_train(trainloader, classes, batch_size)

            for domain, value in single_accuracy.items():
                accuracy[domain] += value

    for domain in accuracy.keys():
        accuracy[domain] /= n_mc_class*n_mc_instance

    return accuracy

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
accuracy_all = {domain: [] for domain in domains}

n_instances = range(2, 21)

t0 = time.time()

for n_instance in n_instances:
    print(n_instance)
    print(time.time() - t0)
    accuracy = train_and_eval(n_instance, 1, 1)
    for domain, acc in accuracy.items():
        accuracy_all[domain].append(acc.item())



colors = ["k", "g", "r", "b", "y"]
plt.figure()
for (domain, values), color in zip(accuracy_all.items(), colors):
    plt.plot(n_instances, values, label=domain, color=color)
plt.legend()
plt.xlabel("Nombre de shots")
plt.ylabel("Précision")
plt.xticks(n_instances, n_instances)
plt.savefig("./accuracies.jpg")
plt.show()
