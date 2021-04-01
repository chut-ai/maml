from maml.meta.data.task_generator import make_tasks
from maml.meta.model import DenseNet
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn.functional as F
import higher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = DenseNet().to(device)
net.train()

print("Making tasks ...")

n_class = 10
n_spt = 50
n_qry = 50
n_task = 1000

tasks = make_tasks("real", n_class, n_spt, n_qry, n_task)

print("Done !")


n_iter_inner_loop = 15
meta_opt = optim.Adam(net.parameters(), lr=0.001)
qry_accs = []

print("Training ...")

for i, task in enumerate(tasks):

    message = "Task {}/{} ({:.1f}%) \r".format(i, len(tasks), 100*i/len(tasks))
    print(message, sep=" ", end="", flush=True)

    x_spt, x_qry, y_spt, y_qry = task
    x_spt = x_spt.to(device)
    y_spt = y_spt.to(device, dtype=torch.int64)
    x_qry = x_qry.to(device)
    y_qry = y_qry.to(device, dtype=torch.int64)

    inner_opt = optim.SGD(net.parameters(), lr=0.1)

    meta_opt.zero_grad()

    with higher.innerloop_ctx(net, inner_opt, device, copy_initial_weights=False) as (fnet, diffopt):
        for _ in range(n_iter_inner_loop):
            spt_logits = fnet(x_spt)
            spt_loss = F.cross_entropy(spt_logits, y_spt)
            diffopt.step(spt_loss)

        qry_logits = fnet(x_qry)
        qry_acc = (qry_logits.argmax(dim=1) ==
                   y_qry).sum().item()/y_qry.size()[0]
        qry_accs.append(qry_acc)
        qry_loss = F.cross_entropy(qry_logits, y_qry)
        qry_loss.backward()
    meta_opt.step()

X = range(len(qry_accs))
plt.figure()
plt.scatter(X, qry_accs)
plt.xlabel("Number of tasks")
plt.ylabel("Accuracy of query")
plt.show()
