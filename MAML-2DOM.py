import argparse
import numpy as np
import torch.optim as optim
from meta.meta import Meta
from meta.graph import draw_acc, draw_loss
from models import DenseNet
from data.task_generator import EncodedVisdaTask

argparser = argparse.ArgumentParser()
argparser.add_argument("--source", type=str,
                       help="Source domain", default="real")
argparser.add_argument("--target", type=str,
                       help="Target domain", default="quickdraw")
args = argparser.parse_args()

source = args.source
target = args.target

n_class = 10  # Nb of classes in classification problem
n_spt = 200  # Nb of source imgs per class
n_qry = 200  # Nb of target imgs per class
task_bsize = 20
n_batch = 200

visda = EncodedVisdaTask(n_class, n_qry, n_spt, [
    source, target], path="./data/json/")

inner_lr = 0.0005
n_inner_loop = 20
net = DenseNet()

meta_model = Meta(net)
meta_model.cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters(), meta_lr)

train_accs = []
train_losses = []
test_accs = []
test_losses = []

for i in range(0, n_batch):

    # Meta train

    meta_opt.zero_grad()
    train_batch = visda.task_batch(task_bsize, "train", source, target)
    train_acc, train_loss = meta_model.train(
        train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    del train_batch
    meta_opt.step()

    # Meta test

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

    message = "Task batch {}/{}, train acc = {:.2f}%, train loss = {:.2f}, test acc = {:.2f}%, test loss = {:.2f}".format(
        i+1, n_batch, 100*train_acc, train_loss, 100*test_acc, test_loss)
    print(message, end="\r")

best_acc = max(test_accs)
print("\n{} -> {} accuracy : {:.3f}".format(source, target, best_acc))
