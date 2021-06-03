import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import get_visda
from data.task_generator import EncodedVisdaTask
from models import DenseNet

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

trainloader, testloader = get_visda(batch_size, domains, ratio, train_class, path="./data/json/")

net = DenseNet()
net.classifier[2] = nn.Linear(256, 200) # Adapt model to a 200 way classification problem to pretrain
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 10

# Pretrain model with train classes on a 200 way classification problem

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
    message = "Pretraining, epoch {}/{}, train acc = {:.2f}, test acc = {:.2f}".format(epoch, n_epochs, train_acc, test_acc)
    print(message, end="\r")
print("")
net.classifier[2] = nn.Linear(256, 10) # Adapt model to a 10 way classification problem

del trainloader
del testloader

n_class = 10
max_qry = 200
max_spt = 200
task_bsize = 10

visda = EncodedVisdaTask(n_class, max_qry, max_spt, [source, target], train_class=train_class, path="./data/json/")

tasks = visda.task_batch(task_bsize, "test", source, target)

for i, task in enumerate(tasks):
    p_net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("Training, task {}/{}".format(i+1, len(tasks)), end="\r")
    x_spt, x_qry, y_spt, y_qry = task
    x_spt, y_spt = x_spt.cuda(), y_spt.cuda().type(torch.int64)
    x_qry, y_qry = x_qry.cuda(), y_qry.cuda().type(torch.int64)
    for epoch in range(1, n_epochs+1):
        
        net.train()
        optimizer.zero_grad()
        y_hat = net(x_spt)
        loss = criterion(y_hat, y_spt)
        loss.backward()
        optimizer.step()
    
    net.eval()
    with torch.no_grad():
        y_hat = net(x_qry)
        acc = torch.eq(y_hat.argmax(dim=1), y_qry).sum().item()/y_qry.size()[0]
    accs.append(acc)

avg = np.mean(accs)
std = np.std(accs)

print("\nPretrained network, source only training, {} -> {} : mean = {:.3f}, std = {:.3f}".format(source, target, avg, std))
