import torch
import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DEFAULT_PATH = "/home/louishemadou/data/VisDA_encoded/"

def open_json(domain, path):
    json_path = path + domain + ".json"

    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()
    return data


class VisdaDataset(Dataset):
    def __init__(self, domains, classes=range(1, 346), path=DEFAULT_PATH):

        self.classes = classes
        self.domains = domains
        
        self.data = {domain: open_json(domain, path) for domain in self.domains} 

        for domain in self.domains:
            keys = self.data[domain].keys()
            for key in range(1, 346):
                if key not in self.classes:
                    self.data[domain].pop(str(key), None)

        self.data_list = []

        for domain in self.domains:
            for key, values in self.data[domain].items():
                for value in values:
                    self.data_list.append((value, int(key)))
        
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        value, label = self.data_list[idx]
        label = self.classes.index(label)
        value = torch.Tensor(value)
        return value, label
