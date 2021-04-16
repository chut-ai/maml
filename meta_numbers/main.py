from maml.meta.data.task_generator import VisdaTask
from maml.meta.model import DenseNet
from maml.meta.meta_train import meta_train
from maml.meta.meta_test import meta_test
from maml.meta.graph import graph
from maml.meta_numbers.data.task_generator import NumberTask
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn.functional as F
import higher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = DenseNet().to(device)
net.train()


n_class = 10
n_spt = 10
n_qry = 100
task_bsize = 50
n_batch = 30
n_iter_inner_loop = 20

print("Loading data ...")

visda = VisdaTask(n_class, n_qry, n_spt)
number = NumberTask(n_qry, n_spt)

print("Done !")

meta_opt = optim.Adam(net.parameters(), lr=0.001)
qry_accs_train = []
qry_accs_test = []

src = "mnist"
tgt = "svhn"

print("Training ...")

for i in range(1, n_batch+1):

    qry_train_acc = meta_train(
        visda, net, meta_opt, n_iter_inner_loop, task_bsize, device)
    qry_accs_train.append(qry_train_acc)

    qry_test_acc = meta_test(number, src, tgt, net, n_iter_inner_loop, task_bsize, device)
    qry_accs_test.append(qry_test_acc)

    message = "Task batch {}/{} ({:.1f}%) \r".format(i, n_batch, 100*i/n_batch)
    print(message, sep=" ", end="", flush=True)

    if i%10 == 0:
        X = range(len(qry_accs_train))
        plt.figure()
        plt.title("Meta learning for domain adaptation, source = {}, target = {}".format(src, tgt))
        plt.scatter(X, qry_accs_train, color="k", label="train")
        plt.scatter(X, qry_accs_test, color="r", label="test")
        plt.legend()
        plt.xlabel("Task batchs")
        plt.ylabel("Accuracy of query")
        img_name = "figures/meta/test.png"
        plt.savefig(img_name)
        plt.close()