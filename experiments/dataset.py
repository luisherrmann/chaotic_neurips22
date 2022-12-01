from typing import Sequence
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_train_data(data_path: str, name: str, transform: Sequence = []):
    if name == "usps":
        transform_list = [transforms.ToTensor()]
        transform_list += [*transform]
        transform = transforms.Compose(transform_list)
        train_data = datasets.USPS(
            data_path, train=True, download=False, transform=transform
        )
    elif name == "fmnist":
        transform_list = [transforms.ToTensor(), transforms.Resize(16)]
        transform_list += [*transform]
        transform = transforms.Compose(transform_list)
        train_data = datasets.FashionMNIST(
            data_path, train=True, download=True, transform=transform
        )
    return train_data


def get_validation_data(data_path: str, name: str, transform: Sequence = []):
    if name == "usps":
        transform_list = [transforms.ToTensor()]
        transform_list += [*transform]
        transform_composition = transforms.Compose(transform_list)
        val_data = datasets.USPS(
            data_path, train=False, download=False, transform=transform_composition
        )
    elif name == "fmnist":
        transform_list = [transforms.ToTensor(), transforms.Resize(16)]
        transform_list += [*transform]
        transform_composition = transforms.Compose(transform_list)
        val_data = datasets.FashionMNIST(
            data_path, train=False, download=True, transform=transform_composition
        )
    return val_data


def prepare_ds(data_path: str, name: str, batch_size: int, debug=False):
    train_data = get_train_data(data_path, name)
    val_data = get_validation_data(data_path, name)
    if debug:
        train_data = Subset(train_data, range(0, 3 * batch_size))
        val_data = Subset(val_data, range(3 * batch_size, 5 * batch_size))
    batch_size_tmp = len(train_data)
    train_loader = DataLoader(
        train_data, batch_size=batch_size_tmp, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size_tmp, shuffle=True, num_workers=2
    )
    batch = next(iter(train_loader))
    x, y = batch
    mean, std = torch.mean(x), torch.std(x)
    transform = [transforms.Normalize(mean, std)]
    train_data = get_train_data(data_path, name, transform=transform)
    val_data = get_validation_data(data_path, name, transform=transform)
    if debug:
        train_data = Subset(train_data, range(0, 3 * batch_size))
        val_data = Subset(val_data, range(3 * batch_size, 5 * batch_size))
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=6
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=6
    )
    return train_loader, val_loader
