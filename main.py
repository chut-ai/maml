import torch.optim as optim
from maml.meta.meta import Meta
from maml.model import DenseNet
from maml.data.encoded_visda import EncodedVisdaTask
from maml.graph import draw_acc, draw_loss

n_class = 10
n_spt = 10
n_qry = 20
task_bsize = 20
n_batch = 50000

visda = EncodedVisdaTask(n_class, n_qry, n_spt)

inner_lr = 0.1
n_inner_loop = 20

net = DenseNet()

meta_model = Meta(net)
meta_model.cuda()
meta_lr = 0.001

meta_opt = optim.Adam(meta_model.parameters())

k_test = 10

train_accs = []
train_losses = []
test_accs = []
test_losses = []

for i in range(0, n_batch):

    meta_opt.zero_grad()
    train_batch = visda.task_batch(task_bsize, "train", "real", "quickdraw")
    train_acc, train_loss = meta_model.train(train_batch, inner_lr, n_inner_loop)
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    meta_opt.step()

    if i % k_test == 0:
        test_batch = visda.task_batch(task_bsize, "test", "real", "quickdraw")
        test_acc, test_loss = meta_model.test(test_batch, inner_lr, n_inner_loop)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
    
    path_acc = "./figures/running_acc.png"
    path_loss = "./figures/running_loss.png"
    title = "MAML-2DOM, real -> quickdraw"
    draw_acc(k_test, train_accs, test_accs, path_acc, title)
    draw_acc(k_test, train_losses, test_losses, path_loss, title)
    
    message = "Task batch {}/{} ({:.1f}%), acc = {:.2f}%, loss = {:.3f}".format(i+1, n_batch, 100*i/n_batch, 100*train_acc, train_loss)
    print(message, end="\r"sh=True)


