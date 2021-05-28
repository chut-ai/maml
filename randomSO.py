import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import DenseNet
from data.task_generator import EncodedVisdaTask

n_epochs = 100
n_class = 10
n_qry = 200
n_spt = 200
task_bsize = 200

source = "real"
target = "infograph"

visda = EncodedVisdaTask(n_class, n_qry, n_spt, [
                         source, target], path="./data/json/")

tasks = visda.task_batch(task_bsize, "test", source, target)

accs = []

for i, task in enumerate(tasks):
    net = DenseNet().cuda()
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
        acc = torch.eq(y_hat.argmax(dim=1), y_qry).sum().item()/y_qry.size()[0]
    accs.append(acc)

avg = np.mean(accs)
std = np.std(accs)

print("{} -> {} : mean = {:.3f}, std = {:.3f}".format(source, target, avg, std))
