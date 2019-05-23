import torch


from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial

def transform_permute(tensor, perm):
    return tensor.index_select(0, perm)

def transform_flatten(tensor):
    return tensor.view(-1, 1).contiguous()


def get_train_valid_loader(data_dir,
                           batch_size,
                           perm,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):

    # define transforms
    all_transform = transforms.Compose([
            transforms.ToTensor(),
            transform_flatten,
            partial(transform_permute, perm=perm)
    ])



    # load the dataset
    dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=all_transform,
    )


    num_train = len(dataset)
    indices = list(range(num_train))
    split = 10000


    train_idx, valid_idx = indices[:-split], indices[-split:]
    train_dataset =  torch.utils.data.Subset(dataset, train_idx)
    valid_dataset =  torch.utils.data.Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    perm,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True):

    # define transform
    all_transform = transforms.Compose([
        transforms.ToTensor(),
        transform_flatten,
        partial(transform_permute, perm=perm)
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=all_transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader