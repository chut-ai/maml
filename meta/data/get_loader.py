from maml.meta.data.dataset import VisdaTask
from torch.utils.data import random_split, DataLoader
from torchvision import transforms


def get_visda(batch_size, num_workers, root, domain, ratio, n_class, n_instance):

    dataset = VisdaTask(root, domain, n_class, n_instance)

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
