from unagi.datasets.celeba.celeba_dataset import CelebA
from unagi.datasets.cifar.cifar_dataset import CIFAR10, CIFAR100
from unagi.datasets.mnist.mnist_dataset import MNIST
from unagi.datasets.tiny_imagenet.tinyimagenet_dataset import TinyImageNet

DATASET_CLASSES = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "cifar10_coarse": CIFAR10,
    "cifar100_coarse": CIFAR100,
    "tinyimagenet": TinyImageNet,
    "tinyimagenet_coarse": TinyImageNet,
    "mnist": MNIST,
    "celeba": CelebA,
}
