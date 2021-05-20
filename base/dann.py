import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from maml.base.models import Encoder, Classifier, Discriminator
from maml.data.encoded_visda import EncodedVisdaTask

f = open("../train_class.txt", "r")
train_class = [int(cls) for cls in f.readlines()]


domain_loss = nn.BCELoss()
class_loss = nn.CrossEntropyLoss()


n_epochs = 100
n_class = 10
max_qry = 500
max_spt = 500
task_bsize = 100


def get_lambda(epoch, n_epochs):
    p = epoch/n_epochs
    return 2. / (1+np.exp(-10.*p)) - 1


visda = EncodedVisdaTask(n_class, max_qry, max_spt, train_class=train_class)

tasks = visda.task_batch(task_bsize, "test", "real", "quickdraw")

n_disc = 10
global_acc = 0

for i, task in enumerate(tasks):
    E = Encoder().cuda()
    C = Classifier().cuda()
    D = Discriminator().cuda()
    E_opt = optim.Adam(E.parameters(), lr=0.001)
    C_opt = optim.Adam(C.parameters(), lr=0.001)
    D_opt = optim.Adam(D.parameters(), lr=0.001)
    print("{}/{}".format(i+1, len(tasks)), end="\r")
    x_spt, x_qry, y_spt, y_qry = task
    x_spt, y_spt = x_spt.cuda(), y_spt.cuda().type(torch.int64)
    x_qry, y_qry = x_qry.cuda(), y_qry.cuda().type(torch.int64)
    n_spt = x_spt.size()[0]
    n_qry = x_qry.size()[0]
    domain_src = torch.ones(n_spt, 1).cuda()
    domain_tgt = torch.zeros(int(n_qry/2), 1).cuda()
    domain_labels = torch.cat([domain_src, domain_tgt], dim=0)
    for epoch in range(1, n_epochs+1):

        E.train()
        C.train()
        D.train()
        # Training discriminator
        x = torch.cat([x_spt, x_qry[:int(n_qry/2)]], dim=0)
        h = E(x)

        for _ in range(n_disc):
            y = D(h.detach())
            Ld = domain_loss(y, domain_labels)
            D.zero_grad()
            Ld.backward()
            D_opt.step()

        c = C(h[:n_spt])
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
    with torch.no_grad():
        x2 = x_qry[int(n_qry/2):]
        y2 = y_qry[int(n_qry/2):]
        y2_hat = C(E(x2))
        acc2 = 100*torch.eq(y2_hat.argmax(dim=1), y2).sum().item()/y2.size()[0]
    global_acc += acc2/len(tasks)

print(global_acc)
