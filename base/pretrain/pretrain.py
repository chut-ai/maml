from maml.base.pretrain.dataloader import get_visda
from maml.base.pretrain.model import DenseNet
import torch
import torch.nn as nn
import torch.optim as optim

f = open("../../train_class.txt", "r")
train_class = [int(cls) for cls in f.readlines()]

batch_size = 64
ratio = 0.8

source = "real"
target = "infograph"

domains = [source, target]


trainloader, testloader = get_visda(batch_size, domains, ratio, train_class)

net = DenseNet()
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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
    message = "Epoch {}/{}, train acc = {:.2f}, test acc = {:.2f}".format(epoch, n_epochs, train_acc, test_acc)
    print(message)

net.classifier[2] = nn.Linear(256, 10)
torch.save(net, "./pt/{}_{}.pt".format(source, target))
