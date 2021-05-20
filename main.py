import pickle
import torch
import torch.optim as optim
from maml.meta.meta import Meta
from maml.model import DenseNet
from maml.data.encoded_visda import EncodedVisdaTask
from maml.data.encoded_office31 import EncodedOfficeTask
from maml.graph import draw_acc, draw_loss

n_class = 10
n_spt = 200
n_qry = 200
task_bsize = 20
n_batch = 2000

f = open("./train_class.txt", "r")
train_class = [int(cls) for cls in f.readlines()]

visda = EncodedVisdaTask(n_class, n_qry, n_spt, train_class=train_class)
office = EncodedOfficeTask(n_class, n_qry, n_spt)

inner_lr = 0.1
n_inner_loop = 10

net = DenseNet()

meta_model = Meta(net)
meta_model.cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters())

k_test = 1

train_accs = []
train_losses = []
test_accs = []
test_losses = []

best_acc = 0
best_model = 0

source = "real"

for i in range(0, n_batch):
    if i % k_test == 0:
        test_batch = office.task_batch(task_bsize, "amazon", "webcam")
        test_acc, test_loss = meta_model.test(test_batch, inner_lr, n_inner_loop)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        del test_batch

    meta_opt.zero_grad()
    train_batch = visda.task_batch(task_bsize, "train")
    train_acc, train_loss = meta_model.train(train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    del train_batch
    meta_opt.step()

    if test_acc > best_acc:
        best_acc = test_acc
        best_model = i
        snapshot_name = "./snapshots/best_maml_all.pt"
        torch.save(meta_model, snapshot_name)

    path_acc = "./figures/running_acc.png"
    path_loss = "./figures/running_loss.png"
    title = "MAML-ALL, amazon -> webcam"
    draw_acc(k_test, train_accs, test_accs, path_acc, title)
    draw_loss(k_test, train_losses, test_losses, path_loss, title)

    message = "Task batch {}/{}, train acc = {:.2f}%, loss = {:.3f}".format(
        i+1, n_batch, 100*train_acc, train_loss)
    print(message, end="\r")

save = True

# if save:
    # dic = {"train_accs": train_accs,
            # "train_losses": train_losses,
            # "test_accs": test_accs,
            # "test_losses": test_losses
            # }
    # name = "{}_{}.pickle".format(source, target)
    # handle = open("./pickle/{}".format(name), "wb")
    # pickle.dump(dic, handle)
