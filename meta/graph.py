import matplotlib.pyplot as plt

def graph(qry_accs_train, qry_accs_test, domains):

    for domain in domains:
        X = range(len(qry_accs_train))
        plt.figure()
        plt.title("Meta learning for domain adaptation, source = {}, target = {}".format("real", domain))
        plt.scatter(X, qry_accs_train, color="k", label="train")
        plt.scatter(X, qry_accs_test[domain], color="r", label="test")
        plt.legend()
        plt.xlabel("Task batchs")
        plt.ylabel("Accuracy of query")
        img_name = "figures/running/{}.png".format(domain)
        plt.savefig(img_name)
