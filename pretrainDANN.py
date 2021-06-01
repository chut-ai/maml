import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.dataloader import get_visda
from models import DenseNet
from models import Discriminator
from maml.data.task_generator import EncodedVisdaTask

argparser = argparse.ArgumentParser()
argparser.add_argument("--source", type=str, help="Source domain", default="real")
argparser.add_argument("--target", type=str, help="Target domain", default="quickdraw")
args = argparser.parse_args()

source = args.source
target = args.target

accs = []

train_class = list(np.random.choice(range(345), 200, replace=False))

batch_size = 64
ratio = 0.8

domains = [source, target]

trainloader, testloader = get_visda(
    batch_size, domains, ratio, train_class, path="./data/json/")

net = DenseNet()  # Encoder + Classifier
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Pretraining classifier & encoder on 200 way classification problem

n_epochs = 10

for epoch in range(1, n_epochs+1):

    train_acc = 0
    net.train()
    for i, (x, y) in enumerate(trainloader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = net(x)
        train_acc += 100*torch.eq(y_hat.argmax(dim=1),
                                  y).sum().item()/(y.size()[0]*len(trainloader))
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

    test_acc = 0
    net.eval()
    with torch.no_grad():
        for (x, y) in testloader:
            x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            test_acc += 100*torch.eq(y_hat.argmax(dim=1),
                                     y).sum().item()/(y.size()[0]*len(testloader))
    message = "Epoch {}/{}, train acc = {:.2f}, test acc = {:.2f}".format(
        epoch, n_epochs, train_acc, test_acc)
    print(message, end="\r")
print("")

net.classifier[2] = nn.Linear(256, 10)

del trainloader
del testloader

domain_loss = nn.BCELoss()
class_loss = nn.CrossEntropyLoss()


n_epochs = 100
n_class = 10
n_qry = 200
n_spt = 200
task_bsize = 10


def get_lambda(epoch, n_epochs):
    p = epoch/n_epochs
    return 2. / (1+np.exp(-10.*p)) - 1


visda = EncodedVisdaTask(n_class, n_qry, n_spt, [
    source, target], path="./data/json/", train_class=train_class)

tasks = visda.task_batch(task_bsize, "test", source, target)

n_disc = 10
accs = []

for i, task in enumerate(tasks):
    E = net.encoder.cuda()
    C = net.classifier.cuda()
    D = Discriminator().cuda()
    E_opt = optim.Adam(E.parameters(), lr=0.001)
    C_opt = optim.Adam(C.parameters(), lr=0.001)
    D_opt = optim.Adam(D.parameters(), lr=0.001)
    print("Training, task {}/{}".format(i+1, len(tasks)), end="\r")
    x_spt, x_qry, y_spt, y_qry = task
    x_spt, y_spt = x_spt.cuda(), y_spt.cuda().type(torch.int64)
    x_qry, y_qry = x_qry.cuda(), y_qry.cuda().type(torch.int64)
    domain_src = torch.ones(x_spt.size(0), 1).cuda()
    domain_tgt = torch.zeros(int(x_qry.size(0)/2), 1).cuda()
    domain_labels = torch.cat([domain_src, domain_tgt], dim=0)
    for epoch in range(1, n_epochs+1):

        E.train()
        C.train()

        # Training discriminator
        x = torch.cat([x_spt, x_qry[:int(x_qry.size(0)/2)]], dim=0)
        h = E(x)

        for _ in range(n_disc):
            y = D(h.detach())
            Ld = domain_loss(y, domain_labels)
            D.zero_grad()
            Ld.backward()
            D_opt.step()

        c = C(h[:x_spt.size(0)])
        y = D(h)
        Lc = class_loss(c, y_spt)
        Ld = domain_loss(y, domain_labels)
        lamda = get_lambda(epoch, n_epochs)
        Ltot = Lc - lamda*Ld

        E.zero_grad()
        C.zero_grad()
        D.zero_grad()

        Ltot.backward()

        C_opt.step()
        E_opt.step()

    E.eval()
    C.eval()
    with torch.no_grad():
        x2 = x_qry[int(x_qry.size(0)/2):]
        y2 = y_qry[int(x_qry.size(0)/2):]
        y2_hat = C(E(x2))
        acc2 = torch.eq(y2_hat.argmax(dim=1), y2).sum().item()/y2.size()[0]
    accs.append(acc2)

avg = np.mean(accs)
std = np.std(accs)

print("\nPretrained network, DANN, {} -> {} : mean = {:.3f}, std = {:.3f}".format(source, target, avg, std))
