import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from maml.meta.meta import Meta
from maml.model import DenseNet, ResNet
from maml.data.task_generator import EncodedVisdaTask
from maml.graph import draw_acc, draw_loss

n_class = 10
n_spt = 200
n_qry = 200
task_bsize = 20
n_batch = 50

source = "real"
target = "infograph"

visda = EncodedVisdaTask(n_class, n_qry, n_spt, [source, target], path="./data/json/")

inner_lr = 0.05
n_inner_loop = 20

net = DenseNet()

meta_model = Meta(net)
meta_model.cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters())

train_accs = []
train_losses = []
test_accs = []
test_losses = []

for i in range(0, n_batch):

    meta_opt.zero_grad()
    train_batch = visda.task_batch(task_bsize, "train", source, target)
    train_acc, train_loss = meta_model.train(train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    del train_batch
    meta_opt.step()

    test_batch = visda.task_batch(task_bsize, "test", source, target)
    test_acc, test_loss = meta_model.test(test_batch, inner_lr, n_inner_loop)
    test_accs.append(test_acc)
    test_losses.append(test_loss)
    del test_batch

    path_acc = "./figures/acc_{}.png".format(target)
    path_loss = "./figures/loss_{}.png".format(target)
    title = "MAML-2DOM, {} -> {}".format(source, target)
    draw_acc(train_accs, test_accs, path_acc, title)
    draw_loss(train_losses, test_losses, path_loss, title)

    message = "Task batch {}/{}, train acc = {:.2f}%, loss = {:.3f}".format(
        i+1, n_batch, 100*train_acc, train_loss)
    print(message, end="\r")

best_acc = max(test_accs)
print("\n{} -> {} accuracy : {:.3f}".format(source, target, best_acc))


# if save:
# dic = {"train_accs": train_accs,
# "train_losses": train_losses,
# "test_accs": test_accs,
# "test_losses": test_losses
# }
# name = "{}_{}.pickle".format(source, target)
# handle = open("./pickle/{}".format(name), "wb")
# pickle.dump(dic, handle)
