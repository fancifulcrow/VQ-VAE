import torch
from torchvision import datasets, transforms


def load_data(root, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    return train_set, test_set
