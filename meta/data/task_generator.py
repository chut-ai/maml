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

def make_one_task(data, n_class, n_spt, n_qry):

    n_instance = n_spt + n_qry
    
    possible_class = []
    for i in range(1, 346):
        if len(data[str(i)]) >= n_instance:
            possible_class.append(i)

    chosen_labels = np.random.choice(possible_class, n_class, False)

    instances_labels = []

    for label in chosen_labels:
        indexes = np.random.choice(
            range(len(data[str(label)])), n_instance, False)
        for index in indexes:
            instance = torch.Tensor(data[str(label)][index])
            instances_labels.append([instance, label])
    random.shuffle(instances_labels)

    instances = [elem[0] for elem in instances_labels]
    labels = squeeze([elem[1] for elem in instances_labels])
    x_spt = torch.stack(instances[:n_spt*n_class], 0)
    x_qry = torch.stack(instances[n_spt*n_class:], 0)
    y_spt = torch.Tensor(labels[:n_spt*n_class])
    y_qry = torch.Tensor(labels[n_spt*n_class:])

    return x_spt, x_qry, y_spt, y_qry

def make_tasks(domain, n_class, n_spt, n_qry, task_bsize, n_batch):
    
    data = open_json(domain)

    tasks = [[make_one_task(data, n_class, n_spt, n_qry) for i in range(task_bsize)] for _ in range(n_batch)]

    return tasks

if __name__ == "__main__":
    data = open_json("real")
    a = make_one_task(data, 10, 50, 50)
