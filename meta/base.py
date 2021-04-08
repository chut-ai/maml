from maml.meta.data.task_generator import VisdaTask
from maml.meta.model import DenseNet
from maml.meta.meta_test import meta_test
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = DenseNet().to(device)
net.train()


n_class = 10
n_spt = 10
n_qry = 40
task_bsize = 50
n_batch = 50
n_iter_inner_loop = 20

print("Loading data ...")

visda = VisdaTask(n_class, n_qry, n_spt)

print("Done !")

domains = ["clipart", "sketch", "quickdraw", "painting", "infograph"]
qry_accs_test = {domain: [] for domain in domains}

print("Training ...")

for i in range(1, n_batch+1):

    for domain in domains:
        qry_test_acc = meta_test(
            visda, "real", domain, net, n_iter_inner_loop, task_bsize, device)
        qry_accs_test[domain].append(qry_test_acc)

    message = "Task batch {}/{} ({:.1f}%) \r".format(i, n_batch, 100*i/n_batch)
    print(message, sep=" ", end="", flush=True)


message = "\nAverage accuracies :\n"

for domain in domains:
    avg_acc = 100*sum(qry_accs_test[domain])/len(qry_accs_test[domain])
    print("{} : {:.3f}%".format(domain, avg_acc))
