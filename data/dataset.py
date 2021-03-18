import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class VisdaDataset(Dataset):
    def __init__(self, root, domain):

        self.domain = domain
        self.imgs_path = "/".join([root, domain])
        self.img_list = os.listdir(self.imgs_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open("/".join([self.imgs_path, img_name]))
        label = int(img_name[-14: -11])
        return img, label


def get_index_to_class():

    with open("/home/louishemadou/dev/maml/data/index_to_class.json") as f:
        index_to_class = json.load(f)
    index_to_class = {int(k): v for k, v in index_to_class.items()}
    return index_to_class


def get_class_to_index():

    with open("./class_to_index.json") as f:
        class_to_index = json.load(f)
    class_to_index = {k: int(v) for k, v in class_to_index.items()}
    return class_to_index
