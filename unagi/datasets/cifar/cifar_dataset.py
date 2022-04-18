import meerkat as mk
import numpy as np
import torch
import torchvision

from unagi.datasets.base_dataset import UnagiDatasetBuilder
from unagi.datasets.cifar.utils import get_superclass_subclass_mapping, sparse2coarse
from unagi.datasets.meerkat_dataset import MeerkatDataset


class CIFAR10(UnagiDatasetBuilder):
    """Dataset to load CIFAR 10 dataset."""

    _name_ = "cifar10"
    # TODO: these can be modified by the transforms (e.g. grayscale)
    # and need to be up to date
    input_shapes = {
        "image": (3, 32, 32),
    }
    output_shapes = {
        "label": (10,),
    }

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,  # For validation split
            "coarse_labels": False,
            "return_train_as_test": False,
            "subset_split_percent": None,
            "subset_split_seed": 42,
        }

    def setup(self):
        self.dataset_train = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
        )
        self.dataset_train, self.dataset_val = self.split_train_val(
            val_split=self.val_split
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
        )
        self.dataset_train = self.to_meerkat(self.dataset_train)
        self.dataset_val = self.to_meerkat(self.dataset_val)
        if self.return_train_as_test:
            self.dataset_test = self.to_meerkat(self.dataset_train)
        else:
            self.dataset_test = self.to_meerkat(self.dataset_test)

    def to_meerkat(self, dataset):
        if self.coarse_labels:
            # TODO: split train and val
            img_pil, label = [], []

            for _, (x, y) in enumerate(dataset):
                img_pil.append(x)
                label.append(y)
            coarse_label = sparse2coarse(label, dataset="cifar10")
            obj = {
                "image": mk.ListColumn(img_pil),
                # "label": mk.TensorColumn(label),
                "label": mk.TensorColumn(coarse_label),
            }
            dp = mk.DataPanel(obj)

            # TODO: combine this with the UnagiDataset as an option
            dataset = MeerkatDataset(dp, xs=["image"], ys=["label"])
            self.output_shapes["label"] = (2,)
        return dataset


class CIFAR10Subset(CIFAR10):
    """Dataset to load subset of CIFAR10 dataset"""

    _name_ = "cifar10_subset"

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "val_split_seed": 42,  # For validation split
            "subset_split_seed": 42,
            "subset_split_percent": 0.5,
            "coarse_labels": False,
        }

    def setup(self):
        super().setup()
        # randomly split the train set
        subset_size = int(len(self.dataset_train) * self.subset_split_percent)
        (dataset_train_subset, dataset_train_heldout,) = torch.utils.data.random_split(
            self.dataset_train,
            (subset_size, len(self.dataset_train) - subset_size),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", self.subset_split_seed)
            ),
        )
        self.dataset_train = dataset_train_subset
        self.dataset_test = {
            "original_testset": self.dataset_test,
            "heldout_trainset": dataset_train_heldout,
        }


class CIFAR100(UnagiDatasetBuilder):
    """Dataset to load CIFAR 100 dataset."""

    _name_ = "cifar100"
    # TODO: these can be modified by the transforms (e.g. grayscale)
    # and need to be up to date
    input_shapes = {
        "image": (3, 32, 32),
    }
    output_shapes = {
        "label": (100,),
    }

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,  # For validation split
            "coarse_labels": False,
            "coarse_labels_u": False,
        }

    def setup(self):
        self.dataset_train = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
        )
        self.dataset_train, self.dataset_val = self.split_train_val(
            val_split=self.val_split
        )
        self.dataset_test = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
        )
        self.dataset_train = self.to_meerkat(self.dataset_train)
        self.dataset_val = self.to_meerkat(self.dataset_val)
        self.dataset_test = self.to_meerkat(self.dataset_test)

    def to_meerkat(self, dataset):
        if self.coarse_labels or self.coarse_labels_u:
            # TODO: split train and val
            img_pil, label = [], []

            for _, (x, y) in enumerate(dataset):
                img_pil.append(x)
                label.append(y)

            if self.coarse_labels_u:
                # img_pil = np.array(img_pil)
                label = np.array(label)

                coarse_u_mapping = get_superclass_subclass_mapping()
                indices = []

                for coarse_label, subclass_labels in coarse_u_mapping.items():
                    total_samples = [500, 250, 100, 50, 50]
                    for subclass_label, samples in zip(subclass_labels, total_samples):
                        indices.extend(
                            np.random.choice(
                                np.argwhere(label == subclass_label).squeeze(-1),
                                size=samples,
                            )
                        )
                temp_img_list = []
                for idx in indices:
                    temp_img_list.append(img_pil[idx])
                img_pil = temp_img_list
                label = label[indices].tolist()

            if self.coarse_labels:
                coarse_label = sparse2coarse(label, dataset="cifar100")
                self.output_shapes["label"] = (20,)
            else:
                coarse_label = label

            obj = {
                "image": mk.ListColumn(img_pil),
                "label": mk.TensorColumn(coarse_label),
            }
            dp = mk.DataPanel(obj)

            # TODO: combine this with the UnagiDataset as an option
            dataset = MeerkatDataset(dp, xs=["image"], ys=["label"])

        return dataset
