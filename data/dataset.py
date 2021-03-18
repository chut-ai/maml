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




dataset = VisdaDataset("/home/louishemadou/VisDA", "real")

with open("./index_to_class.json") as f:
    index_to_class = json.load(f)

for _ in range(20):
    n = np.random.randint(len(dataset))

    img, label = dataset[n]

    label = index_to_class[str(label)]
    print(label)

    plt.figure()
    plt.imshow(img)
    plt.show()
