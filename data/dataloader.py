import json
import random
import torch
from torch.utils.data import random_split, Dataset, DataLoader

DEFAULT_PATH = "./json/"


def open_json(domain, path):
    json_path = path + domain + ".json"

    with open(json_path, "r") as f:
        data = json.load(f)
    f.close()
    return data


class VisdaDataset(Dataset):
    def __init__(self, domains, classes=range(345), path=DEFAULT_PATH):

        self.classes = classes
        self.domains = domains

        self.data = {domain: open_json(domain, path)
                     for domain in self.domains}

        for domain in self.domains:
            keys = self.data[domain].keys()
            for key in range(345):
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


def get_visda(batch_size, domains, ratio, classes=range(345), num_workers=8, path=DEFAULT_PATH):

    dataset = VisdaDataset(domains, classes, path)

    train_size = int(len(dataset)*ratio)
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    if len(trainset) == 0:
        testloader = DataLoader(testset, batch_size, shuffle=True,
                                drop_last=True, num_workers=num_workers)
        return testloader
    elif len(testset) == 0:
        trainloader = DataLoader(trainset, batch_size, shuffle=True,
                                 drop_last=True, num_workers=num_workers)
        return trainloader

    trainloader = DataLoader(
        trainset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers)

    return trainloader, testloader
