import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

DEFAULT_PATH = "/home/louishemadou/data/VisDA_encoded/"


def open_json(domain, path):

    json_path = path + domain + ".json"

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


class EncodedVisdaTask:
    def __init__(self, n_class, n_qry, n_spt,
                 path=DEFAULT_PATH, n_train_class=200):

        self.domains = ["real", "clipart", "infograph",
                        "sketch", "quickdraw", "painting"]
        self.domains = ["real", "quickdraw"]
        self.data = {domain: open_json(domain, path) for domain in self.domains}

        self.n_class = n_class
        self.n_qry = n_qry
        self.n_spt = n_spt
        possible_class = []
        for i in range(1, 346):
            if min([len(self.data[domain][str(i)]) for domain in self.domains]) >= n_spt:
                possible_class.append(i)

        n_test_class = n_train_class - len(possible_class)

        self.train_class = list(np.random.choice(
            possible_class, n_train_class, False))
        self.test_class = [
            x for x in possible_class if x not in self.train_class]

    def task(self, task_classes, source=None, target=None):

        chosen_labels = np.random.choice(task_classes, self.n_class, False)

        if source is None:
            id1, id2 = list(np.random.choice(len(self.domains), 2, False))
            source = self.domains[id1]
            target = self.domains[id2]

        spt_data = []
        for label in chosen_labels:
            all_instances = self.data[source][str(label)]
            indexes = np.random.choice(len(all_instances), self.n_spt, False)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
                spt_data.append([instance, label])
        random.shuffle(spt_data)

        qry_data = []
        for label in chosen_labels:
            all_instances = self.data[target][str(label)]
            n_img = len(all_instances)
            if self.n_qry > n_img:
                indexes = range(n_img)
            else:
                indexes = np.random.choice(n_img, self.n_qry, False)
            for index in indexes:
                instance = torch.Tensor(all_instances[index])
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

    def task_batch(self, task_bsize, mode, source=None, target=None):

        if mode == "train":
            task_classes = self.train_class
        elif mode == "test":
            task_classes = self.test_class
        else:
            raise "WrongModeError"

        tasks = [self.task(task_classes, source, target)
                 for _ in range(task_bsize)]

        return tasks
