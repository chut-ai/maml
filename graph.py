import matplotlib.pyplot as plt
import numpy as np
import torch

def draw_acc(k_fig, train_accs, test_accs, path, title):

    X1 = range(len(train_accs))
    X2 = range(0, k_fig*len(test_accs), k_fig)
    plt.figure()
    plt.scatter(X1, train_accs, color="k", label="train")
    plt.scatter(X2, test_accs, color="r", label="test")
    plt.legend()
    plt.grid()
    plt.xlabel("Task batchs")
    plt.ylabel("Query accuracy")
    plt.title(title)
    plt.savefig(path)
    plt.close()

def draw_loss(k_fig, train_losses, test_losses, path, title):

    X1 = range(len(train_losses))
    X2 = range(0, k_fig*len(test_losses), k_fig)
    plt.figure()
    plt.plot(X1, train_losses, color="k", label="train")
    plt.plot(X2, test_losses, color="r", label="test")
    plt.ylim(0, 3)
    plt.legend()
    plt.xlabel("Task batchs")
    plt.ylabel("Query loss")
    plt.title(title)
    plt.savefig(path)
    plt.close()
