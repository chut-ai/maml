import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from maml.data.get_loader import get_visda
from maml.office31.base.model import DenseNet
from maml.office31.base.data import EncodedVisdaDataset

batch_size = 64
n_class = 345

classes = list(np.random.choice(range(1, 346), n_class, replace=False))

domains = ["real", "quickdraw", "painting", "clipart", "infograph", "sketch"]

dataset = EncodedVisdaDataset(domains, classes)
n_train = int(0.7*len(dataset))
n_test = len(dataset) - n_train
trainset, testset = random_split(dataset, [n_train, n_test])

trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classifier = DenseNet()
classifier.to(device)

n_epoch = 10
optimizer = optim.Adam(classifier.parameters(), lr=0.002, betas=(0.9, 0.999))
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
            out = classifier(val_imgs)
            predicted = torch.max(out, 1)[1].to(device)
            correct += (predicted == val_labels).sum()
        correct_percent = 100*correct/(len(loader)*batch_size)
    return correct_percent


for epoch in range(n_epoch):

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        classifier.train()

        optimizer.zero_grad()
        out = classifier(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        message = "Epoch {}/{}, ({:.0f}%) \r".format(epoch+1,
                                                     n_epoch, 100*i/len(trainloader))
        print(message, sep=" ", end="", flush=True)

    classifier.eval()
    
    train_score = accuracy(trainloader)
    test_score = accuracy(testloader)

    acc_list_test.append(test_score.cpu())
    acc_list_train.append(train_score.cpu())
    elapsed = (time.time()-T0)/60
    print("\rEpoch {}, test accuracy: {:.3f}, train accuracy: {:.3f}, {:.0f} minutes elapsed".format(epoch+1, test_score, train_score, elapsed))

test_score = accuracy(testloader)
train_score = accuracy(trainloader)

print("train score {} test score {}".format(train_score, test_score))

torch.save(classifier, "./saved/classifier_visda")



X = range(len(acc_list_test))

plt.figure(1)
plt.plot(X, acc_list_test, "k", label="test score")
plt.plot(X, acc_list_train, "r", label="train score")
plt.legend()
plt.title("Accuracy over epochs")
plt.show()
