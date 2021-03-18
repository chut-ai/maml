from maml.data.dataset import VisdaDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms


def get_visda(batch_size, num_workers, root, domain, ratio, crop=400, size=150):

    dataset = VisdaDataset(root, domain, crop, size)

    train_size = int(len(dataset)*ratio)
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(
        trainset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers)

    return trainloader, testloader
