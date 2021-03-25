import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
from maml.data.get_loader import get_visda
from maml.base.normal.modelimport ResNet18

batch_size = 32
n_class_train = 200
n_class = 10

classes = list(np.random.choice(range(344), n_class, replace=False))

trainloader, testloader = get_visda(
    batch_size, 8, "/home/louishemadou/VisDA", "real", 0.7, classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = ResNet18()
encoder.to(device)
encoder.eval()

classifier = torch.load("./classifier_visda")
num_features = classifier.layers[6].in_features

classifier.layers[6] = nn.Linear(num_features, n_class)
classifier.to(device)

n_epoch = 25
optimizer = optim.Adam(classifier.parameters(), lr=0.005, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

acc_list_train = []
acc_list_test = []

T0 = time.time()


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

        message = "Epoch {}/{} ({:.0f}%) \r".format(epoch+1,
                                                    n_epoch, 100*i/len(trainloader))
        print(message, sep=" ", end="", flush=True)

    classifier.eval()

    test_score = accuracy(testloader)
    train_score = accuracy(trainloader)

    acc_list_test.append(test_score.cpu())
    acc_list_train.append(train_score.cpu())
    elapsed = (time.time()-T0)/60
    print("\rEpoch {}, test accuracy: {:.3f}, train accuracy: {:.3f}, {:.0f} minutes elapsed".format(
        epoch+1, test_score, train_score, elapsed))


classifier.eval()

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]

for domain in domains:

    _, target_test = get_visda(
        batch_size, 8, "/home/louishemadou/VisDA", "clipart", 0.7, classes)

    target_correct_percent = accuracy(target_test)

    print("{} accuracy : {:.3f}".format(domain, target_correct_percent))
