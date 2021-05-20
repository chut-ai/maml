import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from maml.data.encoded_visda import EncodedVisdaTask

f = open("../../train_class.txt", "r")
train_class = [int(cls) for cls in f.readlines()]


domain_loss = nn.BCELoss()
class_loss = nn.CrossEntropyLoss()


n_epochs = 100
n_class = 10
max_qry = 500
max_spt = 500
task_bsize = 100


visda = EncodedVisdaTask(n_class, max_qry, max_spt, train_class=train_class)

source = "real"
target = "quickdraw"

tasks = visda.task_batch(task_bsize, "test", source, target)

global_acc = 0

for i, task in enumerate(tasks):
    net = torch.load("./pt/{}_{}.pt".format(source, target)).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("{}/{}".format(i+1, len(tasks)), end="\r")
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
        acc = 100*torch.eq(y_hat.argmax(dim=1), y_qry).sum().item()/y_qry.size()[0]
    global_acc += acc/len(tasks)

print(global_acc)
