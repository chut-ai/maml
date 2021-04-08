import torch
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class VisdaDataset(Dataset):
    def __init__(self, root, domains, labels=range(345), n_instances=-1):

        self.labels = labels
        self.domains = domains
        self.imgs_path = ["/".join([root, domain]) for domain in self.domains]
        
        all_img_list = []

        for path in self.imgs_path:
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    all_img_list.append(os.path.join(dirpath, f))

        random.shuffle(all_img_list)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, (1, 1), (1, 1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.img_list = [
            img_name for img_name in all_img_list if self.get_label(img_name) in labels]
        if n_instances != -1:
            self.img_list = self.sample(n_instances)

    def sample(self, n_instances):
        
        img_list = []
        label_to_img = {label: [] for label in self.labels}
        for img in self.img_list:
            label = self.get_label(img)
            label_to_img[label].append(img)

        for label, imgs in label_to_img.items():
            choice = np.random.choice(imgs, n_instances, replace=False)
            img_list += list(choice)
        return img_list



    def get_label(self, img_name, squeeze=False):
        
        label = int(img_name[-14: -11])
        if label == 345:
            label = 0
        if squeeze:
            label = self.labels.index(label)
        return label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(img_name)
        img = self.transform(img)
        label = self.get_label(img_name, True)
        return img, label


def get_index_to_label():

    with open("/home/louishemadou/dev/maml/data/index_to_class.json") as f:
        index_to_label = json.load(f)
    index_to_label = {int(k): v for k, v in index_to_label.items()}
    return index_to_label


def get_label_to_index():

    with open("./class_to_index.json") as f:
        label_to_index = json.load(f)
    label_to_index = {k: int(v) for k, v in label_to_index.items()}
    return label_to_index
