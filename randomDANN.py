import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import DenseNet, Discriminator
from data.task_generator import EncodedVisdaTask

argparser = argparse.ArgumentParser()
argparser.add_argument("--source", type=str, help="Source domain", default="real")
argparser.add_argument("--target", type=str, help="Target domain", default="quickdraw")
args = argparser.parse_args()

source = args.source
target = args.target

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

visda = EncodedVisdaTask(n_class, n_qry, n_spt, [source, target], "./data/json/")

tasks = visda.task_batch(task_bsize, "test", source, target)

n_disc = 10
accs = []

for i, task in enumerate(tasks):
    net = DenseNet()
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
        D.train()

        # Training discriminator to optimality

        x = torch.cat([x_spt, x_qry[:int(x_qry.size(0)/2)]], dim=0)
        h = E(x)
        for _ in range(n_disc):
            y = D(h.detach())
            Ld = domain_loss(y, domain_labels)
            D.zero_grad()
            Ld.backward()
            D_opt.step()

        # Training encoder & classifier

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
    C.eval()
    E.eval()
    with torch.no_grad():
        x2 = x_qry[int(x_qry.size(0)/2):]
        y2 = y_qry[int(x_qry.size(0)/2):]
        y2_hat = C(E(x2))
        acc2 = torch.eq(y2_hat.argmax(dim=1), y2).sum().item()/y2.size()[0]
    accs.append(acc2)

avg = np.mean(accs)
std = np.std(accs)

print("\nRandom network, DANN, {} -> {} : mean = {:.3f}, std = {:.3f}".format(source, target, avg, std))
