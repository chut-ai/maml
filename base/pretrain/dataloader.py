from maml.base.pretrain.dataset import VisdaDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

DEFAULT_PATH = "/home/louishemadou/data/VisDA_encoded/"

def get_visda(batch_size, domains, ratio, classes=range(1, 346), num_workers=8, path=DEFAULT_PATH):

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
