import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def open_json(domain):

    json_path = "/home/louishemadou/VisDA_encoded/" + domain + ".json"

    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()
    return data


def squeeze(labels):

    items = []
    for label in labels:
        if label not in items:
            items.append(label)

    items = sorted(items)

    squeezed = []
    for label in labels:
        squeezed.append(items.index(label))
    return squeezed




class VisdaTask:
    def __init__(self, source, target, n_class, n_qry, n_spt):
        self.source = source
        self.target = target
        self.data_src = open_json(self.source)
        self.data_tgt = open_json(self.target)

        self.n_class = n_class
        self.n_qry = n_qry
        self.n_spt = n_spt
        possible_class = []
        for i in range(1, 346):
            if min(len(self.data_src[str(i)]), len(self.data_tgt[str(i)])) >= max(n_qry, n_spt):
                possible_class.append(i)
        n_test_class = int(len(possible_class)/2)
        n_train_class = len(possible_class) - n_test_class

        self.train_class = list(np.random.choice(
            possible_class, n_train_class, False))
        self.test_class = [
            x for x in possible_class if x not in self.train_class]

    def task(self, mode):

        if mode == "train":
            class_list = self.train_class
        elif mode == "test":
            class_list = self.test_class
        else:
            print('mode has to be "train" or "test"')

        chosen_labels = np.random.choice(class_list, self.n_class, False)

        spt_data = []
        for label in chosen_labels:
            indexes = np.random.choice(
                range(len(self.data_src[str(label)])), self.n_spt, False)
            for index in indexes:
                instance = torch.Tensor(self.data_src[str(label)][index])
                spt_data.append([instance, label])
        random.shuffle(spt_data)
        
        qry_data = []
        for label in chosen_labels:
            indexes = np.random.choice(
                range(len(self.data_tgt[str(label)])), self.n_qry, False)
            for index in indexes:
                instance = torch.Tensor(self.data_tgt[str(label)][index])
                qry_data.append([instance, label])
        random.shuffle(qry_data)
        
        instances_spt = [elem[0] for elem in spt_data]
        labels_spt = squeeze([elem[1] for elem in spt_data])

        instances_qry = [elem[0] for elem in qry_data]
        labels_qry = squeeze([elem[1] for elem in qry_data])
       
        x_spt = torch.stack(instances_spt, 0)
        x_qry = torch.stack(instances_qry, 0)
        y_spt = torch.Tensor(labels_spt)
        y_qry = torch.Tensor(labels_qry)

        return x_spt, x_qry, y_spt, y_qry

    def task_batch(self, mode, task_bsize):

        tasks = [self.task(mode) for _ in range(task_bsize)]

        return tasks


if __name__ == "__main__":
    visdatask = VisdaTask("clipart", 10, 10, 10)
    task = visdatask.task_batch("train", 10)
    print(len(task[0]))
