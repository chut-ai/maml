from maml.meta.data.task_generator import VisdaTask
from maml.office31.model import DenseNet
from maml.meta.meta_train import meta_train
from maml.meta.meta_test import meta_test
from maml.meta.graph import graph
from maml.office31.data.task_generator import OfficeTask
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
bs = 50  # Task batch size
n_batch = 10000
n_inner_loop = 20

print("Loading data ...")

visda = VisdaTask(n_class, n_qry, n_spt)
office = OfficeTask(n_qry, n_spt)

print("Done !")

meta_opt = optim.Adam(net.parameters(), lr=0.001)
qry_accs_train = []

source = "amazon"
target = "dslr"
qry_accs_test = []

print("Training ...")

for i in range(1, n_batch+1):

    # Train with visda domain shifts
    qry_train_acc = meta_train(visda, net, meta_opt, n_inner_loop, bs, device)
    qry_accs_train.append(qry_train_acc)

    # Test on office31 domain shifts

    acc = meta_test(office, source, target, net, n_inner_loop, bs, device)
    qry_accs_test.append(acc)

    message = "Task batch {}/{} ({:.1f}%) \r".format(i, n_batch, 100*i/n_batch)
    print(message, sep=" ", end="", flush=True)

    if i % 10 == 0:
        X = range(len(qry_accs_train))
        plt.figure()
        plt.title("Meta training on visda, meta testing on source = {}, target = {}".format(
            source, target))
        plt.scatter(X, qry_accs_train, color="k", label="train")
        plt.scatter(X, qry_accs_test, color="r", label="test")
        plt.legend()
        plt.xlabel("Task batchs")
        plt.ylabel("Accuracy of query")
        img_name = "figures/meta/test.png"
        plt.savefig(img_name)
