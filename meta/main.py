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
n_spt = 10
n_qry = 90
task_bsize = 10
n_batch = 500

tasks = make_tasks("real", n_class, n_spt, n_qry, task_bsize, n_batch)

print("Done !")

n_iter_inner_loop = 20
meta_opt = optim.Adam(net.parameters(), lr=0.001)
qry_accs = []

print("Training ...")

for i, task_batch in enumerate(tasks):

    message = "Task batch {}/{} ({:.1f}%) \r".format(i, len(tasks), 100*i/len(tasks))
    print(message, sep=" ", end="", flush=True)


    inner_opt = optim.SGD(net.parameters(), lr=0.1)

    task_qry_acc = 0
    meta_opt.zero_grad()

    for task in task_batch:
        x_spt, x_qry, y_spt, y_qry = task
        x_spt = x_spt.to(device)
        y_spt = y_spt.to(device, dtype=torch.int64)
        x_qry = x_qry.to(device)
        y_qry = y_qry.to(device, dtype=torch.int64)
        

        with higher.innerloop_ctx(net, inner_opt, device, copy_initial_weights=False) as (fnet, diffopt):
            for _ in range(n_iter_inner_loop):
                spt_logits = fnet(x_spt)
                spt_loss = F.cross_entropy(spt_logits, y_spt)
                diffopt.step(spt_loss)

            qry_logits = fnet(x_qry)
            task_qry_acc += (qry_logits.argmax(dim=1) ==
                    y_qry).sum().item()/(y_qry.size()[0]*len(task_batch))
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_loss.backward()
    qry_accs.append(task_qry_acc)
    meta_opt.step()

X = range(len(qry_accs))
plt.figure()
plt.scatter(X, qry_accs)
plt.xlabel("Number of tasks")
plt.ylabel("Accuracy of query")
plt.show()
