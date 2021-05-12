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

database = "VisDA"
domain = "painting"
resolution = 224

transform = transforms.RandomResizedCrop(resolution, (1, 1), (1, 1))
path = "/home/louishemadou/data/{}/{}/".format(database, domain)
new_path = "/home/louishemadou/data/{}{}/{}/".format(database, resolution, domain)
img_list = os.listdir(path)
n_img = len(img_list)

for i, img_name in enumerate(img_list):
    print("{}/{}".format(i+1, n_img), end="\r")
    img = Image.open(path+img_name)
    new_img = transform(img)
    new_img_path = new_path + img_name
    new_img.save(new_img_path)
