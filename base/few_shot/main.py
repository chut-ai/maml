import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from maml.data.get_loader import get_visda
from maml.base.few_shot.train import single_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_and_eval(n_instance, n_mc_class, n_mc_instance):

    n_class = 10
    batch_size = n_instance

    domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
    accuracy = {domain: 0 for domain in domains}

    for i in range(n_mc_class):
        classes = list(np.random.choice(range(345), n_class, replace=False))

        for j in range(n_mc_instance):

            trainloader = get_visda(
                batch_size, 8, "/home/louishemadou/VisDA", "real", 1, classes, n_instance)

            single_accuracy = single_train(trainloader, classes, batch_size, device)

            for domain, value in single_accuracy.items():
                accuracy[domain] += value

    for domain in accuracy.keys():
        accuracy[domain] /= n_mc_class*n_mc_instance

    return accuracy

domains = ["clipart", "quickdraw", "painting", "sketch", "infograph"]
accuracy_all = {domain: [] for domain in domains}

n_instances = range(2, 21)

t0 = time.time()

for n_instance in n_instances:
    print(n_instance)
    print(time.time() - t0)
    accuracy = train_and_eval(n_instance, 1, 1)
    for domain, acc in accuracy.items():
        accuracy_all[domain].append(acc.item())



colors = ["k", "g", "r", "b", "y"]
plt.figure()
for (domain, values), color in zip(accuracy_all.items(), colors):
    plt.plot(n_instances, values, label=domain, color=color)
plt.legend()
plt.xlabel("Nombre de shots")
plt.ylabel("Précision")
plt.xticks(n_instances, n_instances)
plt.savefig("./accuracies.jpg")
plt.show()
