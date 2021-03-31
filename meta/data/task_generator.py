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


def make_one_task(data, n_class, n_instance):


    chosen_labels = np.random.choice(range(1, 346), n_class, False)

    instances_labels = []

    for label in chosen_labels:
        indexes = np.random.choice(
            range(len(data[str(label)])), n_instance, False)
        for index in indexes:
            instance = torch.Tensor(data[str(label)][index])
            instances_labels.append([instance, label])
    random.shuffle(instances_labels)

    instances = [elem[0] for elem in instances_labels]
    x = torch.stack(instances, 0)
    labels = [elem[1] for elem in instances_labels]
    y = torch.Tensor(labels)

    return x, y

def make_tasks(domain, n_class, n_instance, n_task):
    
    data = open_json(domain)

    tasks = [make_one_task(data, n_class, n_instance) for _ in range(n_task)]

    return tasks

if __name__ == "__main__":
    domain = "clipart"
    t0 = time.time()
    tasks = make_tasks(domain, 10, 10, 1000)
    print(time.time()-t0)
