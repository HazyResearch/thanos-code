import meerkat as mk
import torchvision

from unagi.datasets.base_dataset import UnagiDatasetBuilder
from unagi.datasets.meerkat_dataset import MeerkatDataset
from unagi.datasets.mnist.utils import sparse2coarse


class MNIST(UnagiDatasetBuilder):
    """Dataset to load MNIST dataset."""

    _name_ = "mnist"
    # TODO: these can be modified by the transforms (e.g. grayscale)
    # and need to be up to date
    input_shapes = {
        "image": (1, 28, 28),
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
        }

    def setup(self):
        self.dataset_train = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
        )
        self.dataset_train, self.dataset_val = self.split_train_val(
            val_split=self.val_split
        )
        self.dataset_test = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
        )
        self.dataset_train = self.to_meerkat(self.dataset_train)
        self.dataset_val = self.to_meerkat(self.dataset_val)
        self.dataset_test = self.to_meerkat(self.dataset_test)

    def to_meerkat(self, dataset):
        if self.coarse_labels:
            # TODO: split train and val
            img_pil, label = [], []

            for _, (x, y) in enumerate(dataset):
                img_pil.append(x)
                label.append(y)
            coarse_label = sparse2coarse(label, dataset="mnist")
            obj = {
                "image": mk.ListColumn(img_pil),
                "label": mk.TensorColumn(coarse_label),
            }
            dp = mk.DataPanel(obj)

            # TODO: combine this with the UnagiDataset as an option
            dataset = MeerkatDataset(dp, xs=["image"], ys=["label"])
            self.output_shapes["label"] = (2,)
        return dataset
