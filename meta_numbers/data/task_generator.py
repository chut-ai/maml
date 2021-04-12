import json
import numpy as np
import random
import torch

class NumberTask:
    def __init__(self, n_qry, n_spt):
        with open("/home/louishemadou/MNIST_encoded/mnist.json", "r") as f1:
            self.mnist = json.load(f1)
        f1.close()
        with open("/home/louishemadou/SVHN_encoded/svhn.json", "r") as f2:
            self.svhn = json.load(f2)
        f2.close()
        self.n_qry = n_qry
        self.n_spt = n_spt

    def task(self, src, tgt):

        if src == "mnist" and tgt == "svhn":
            src_data = self.mnist
            tgt_data = self.svhn
        elif tgt == "mnist" and src == "svhn":
            tgt_data = self.mnist
            src_data = self.svhn
        else:
            print("Wrong domains")
        
        spt_data = []
        labels = range(10)
        for label in labels:
            all_instances = src_data[str(label)]
            indexes = np.random.choice(len(all_instances), self.n_spt, False)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
                spt_data.append([instance, label])
        random.shuffle(spt_data)

        qry_data = []
        for label in labels:
            all_instances = tgt_data[str(label)]
            indexes = np.random.choice(len(all_instances), self.n_qry, False)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
                qry_data.append([instance, label])
        random.shuffle(qry_data)
        
        instances_spt = [elem[0] for elem in spt_data]
        labels_spt = [elem[1] for elem in spt_data]

        instances_qry = [elem[0] for elem in qry_data]
        labels_qry = [elem[1] for elem in qry_data]

        x_spt = torch.stack(instances_spt, 0)
        x_qry = torch.stack(instances_qry, 0)
        y_spt = torch.Tensor(labels_spt)
        y_qry = torch.Tensor(labels_qry)

        return x_spt, x_qry, y_spt, y_qry

    def test_task_batch(self, src, tgt, task_bsize):

        tasks = [self.task(src, tgt) for _ in range(task_bsize)]

        return tasks
