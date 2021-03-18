from maml.data.dataset import VisdaDataset, get_index_to_class
from maml.data.get_loader import get_visda
from torchvision import transforms
import json
import numpy as np
import matplotlib.pyplot as plt

# dataset = VisdaDataset("/home/louishemadou/VisDA", "painting")

# index_to_class = get_index_to_class()

# for _ in range(20):
    # n = np.random.randint(len(dataset))

    # img, label = dataset[n]

    # label = index_to_class[label]
    # print(label)

    # plt.figure()
    # plt.imshow(img)
    # plt.show()


trainloader, testloader = get_visda(64, 8, "/home/louishemadou/VisDA", "quickdraw", 0.8, 150)

print(next(iter(trainloader)))
