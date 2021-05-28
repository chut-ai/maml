import matplotlib.pyplot as plt

def draw_acc(train_accs, test_accs, path, title):

    X1 = range(len(train_accs))
    X2 = range(len(test_accs))
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

def draw_loss(train_losses, test_losses, path, title):

    X1 = range(len(train_losses))
    X2 = range(len(test_losses))
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
