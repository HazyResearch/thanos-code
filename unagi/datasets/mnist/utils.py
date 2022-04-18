import numpy as np


def sparse2coarse(targets, scramble=False, dataset="mnist"):
    """Convert Pytorch MNIST sparse targets.
    trainset = torchvision.datasets.CIFAR100(path)
    trainset.targets = sparse2coarse(trainset.targets)
    """
    if dataset == "mnist":
        sparse_coarse_array = [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ]

    targets = np.array(sparse_coarse_array)[targets]
    return targets.tolist()
