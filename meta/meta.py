import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import higher
import time

def L2_loss(net, fnet, lambdas):
    L = 0
    for p1, p2, lamb in zip(net.parameters(), fnet.parameters(), lambdas):
        L += (torch.exp(-lamb)*(p1-p2)**2).sum()
    return L

class Meta(nn.Module):
    def __init__(self, net):
        super(Meta, self).__init__()
        self.net = net

    def train(self, task_batch, inner_lr, n_inner_loop):

        self.net.train()

        inner_opt = optim.SGD(self.net.parameters(), lr=inner_lr)


        qry_acc = 0

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry = task
            x_spt, y_spt = x_spt.cuda(), y_spt.cuda().type(torch.int64)
            x_qry, y_qry = x_qry.cuda(), y_qry.cuda().type(torch.int64)

            with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits = fnet(x_spt)
                    spt_loss = F.cross_entropy(spt_logits, y_spt)
                    diffopt.step(spt_loss)

                qry_logits = fnet(x_qry)
                qry_loss = F.cross_entropy(qry_logits, y_qry)
                qry_loss.backward()

                qry_logits = fnet(x_qry).detach()
                qry_acc += (qry_logits.argmax(dim=1) ==
                            y_qry).sum().item()/(y_qry.size()[0]*len(task_batch))

        return qry_acc, qry_loss.cpu().detach().numpy().item()

    def test(self, task_batch, inner_lr, n_inner_loop):

        self.net.train()

        inner_opt = optim.SGD(self.net.parameters(), lr=inner_lr)

        qry_acc = 0

        for task in task_batch:
            x_spt, x_qry, y_spt, y_qry = task
            x_spt, y_spt = x_spt.cuda(), y_spt.cuda().type(torch.int64)
            x_qry, y_qry = x_qry.cuda(), y_qry.cuda().type(torch.int64)

            with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(n_inner_loop):
                    spt_logits = fnet(x_spt)
                    spt_loss = F.cross_entropy(spt_logits, y_spt)
                    diffopt.step(spt_loss)

                qry_logits = fnet(x_qry).detach()
                qry_loss = F.cross_entropy(qry_logits, y_qry)
                qry_acc += (qry_logits.argmax(dim=1) ==
                            y_qry).sum().item()/(y_qry.size()[0]*len(task_batch))

        return qry_acc, qry_loss.cpu().detach().numpy().item()
