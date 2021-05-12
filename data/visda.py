import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

DEFAULT_PATH = "/home/louishemadou/data/"


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


def get_all_data(db_name, path):

    all_data = {}

    domains = ["real", "clipart", "infograph",
               "sketch", "quickdraw", "painting"]
    for domain in domains:
        domain_data = {idx: [] for idx in range(1, 346)}
        data_path = "{}{}/{}".format(path, db_name, domain)
        for img_path in os.listdir(data_path):
            label = int(img_path[-14: -11])
            domain_data[label].append(img_path)
        all_data[domain] = domain_data

    return all_data


class VisdaTask:
    def __init__(self, resolution, n_class, n_qry, n_spt,
                 path=DEFAULT_PATH, train_class=None):

        self.domains = ["real", "clipart", "infograph",
                        "sketch", "quickdraw", "painting"]

        self.resolution = resolution
        self.path = path
        self.db_name = "VisDA{}".format(resolution)
        self.data = get_all_data(self.db_name, path)

        self.n_class = n_class
        self.n_qry = n_qry
        self.n_spt = n_spt
        possible_class = []

        for i in range(1, 346):
            min_shot = min([len(self.data[domain][i])
                            for domain in self.domains])
            if min_shot >= n_spt:
                possible_class.append(i)
        
        if train_class is None:
            self.train_class = list(np.random.choice(
                possible_class, 200, False))
        else:
            self.train_class = train_class
        
        self.test_class = [
            x for x in possible_class if x not in self.train_class]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def task(self, task_classes, source=None, target=None, augment=False):

        chosen_classes = list(np.random.choice(
            task_classes, self.n_class, False))

        if source is None:
            id1, id2 = list(np.random.choice(
                range(len(self.domains)), 2, False))

            source = self.domains[id1]
            target = self.domains[id2]

        source_path = "{}{}/{}/".format(self.path, self.db_name, source)
        source_imgs_all = {idx: [] for idx in chosen_classes}
        target_path = "{}{}/{}/".format(self.path, self.db_name, target)
        target_imgs_all = {idx: [] for idx in chosen_classes}

        for img in os.listdir(source_path):
            if (label := int(img[-14: -11])) in chosen_classes:
                img_path = source_path + img
                source_imgs_all[label].append(img_path)

        for img in os.listdir(target_path):
            if (label := int(img[-14: -11])) in chosen_classes:
                img_path = target_path + img
                target_imgs_all[label].append(img_path)

        source_imgs_path = []
        target_imgs_path = []

        for imgs in source_imgs_all.values():
            chosen_imgs = list(np.random.choice(imgs, self.n_spt, False))
            source_imgs_path += chosen_imgs

        for imgs in source_imgs_all.values():
            if len(imgs) < self.n_qry:
                chosen_imgs = imgs
            else:
                chosen_imgs = list(np.random.choice(imgs, self.n_qry, False))
            target_imgs_path += chosen_imgs

        random.shuffle(source_imgs_path)
        random.shuffle(target_imgs_path)

        x_spt = []
        y_spt = []
        x_qry = []
        y_qry = []

        for path in source_imgs_path:
            img = Image.open(path)
            if augment:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.resolution, (0.6, 1.)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((-30, 30)),
                    transforms.ColorJitter(0.2, 0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                img = transform(img)
            else:
                img = self.transform(img)
            label = chosen_classes.index(int(path[-14: -11]))
            x_spt.append(img)
            y_spt.append(label)

        for path in target_imgs_path:
            img = Image.open(path)
            img = self.transform(img)
            label = chosen_classes.index(int(path[-14: -11]))
            x_qry.append(img)
            y_qry.append(label)

        x_spt = torch.stack(x_spt, 0)
        x_qry = torch.stack(x_qry, 0)
        y_spt = torch.Tensor(y_spt)
        y_qry = torch.Tensor(y_qry)

        return x_spt, x_qry, y_spt, y_qry

    def task_batch(self, task_bsize, mode, source=None, target=None, augment=False):

        if mode == "train":
            task_classes = self.train_class
        elif mode == "test":
            task_classes = self.test_class
        else:
            raise "WrongModeError"
        tasks = [self.task(task_classes, source, target, augment)
                 for _ in range(task_bsize)]
        return tasks
# visda = VisdaTask(64, 10, 20, 10)

# tasks = visda.task_batch(64, "train", augment=True)
