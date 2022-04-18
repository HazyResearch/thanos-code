import os

import meerkat as mk
import torchvision

from unagi.datasets.base_dataset import UnagiDatasetBuilder
from unagi.datasets.meerkat_dataset import MeerkatDataset
from unagi.datasets.tiny_imagenet.utils import create_val_img_folder, sparse2coarse


class TinyImageNet(UnagiDatasetBuilder):
    """Dataset to load TinyImageNet dataset."""

    _name_ = "tinyimagenet"
    # TODO: these can be modified by the transforms (e.g. grayscale)
    # and need to be up to date
    input_shapes = {
        "image": (3, 64, 64),
    }
    output_shapes = {
        "label": (200,),
    }

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,  # For validation split
            "coarse_labels": False,
            "root_folder": None,
        }

    def setup(self):
        if self.root_folder is None:
            raise Exception(
                "Please specify the path to root folder containing " "TinyImageNet"
            )

        dp = {"train": {}, "val": {}}

        for split in ["train", "val"]:
            if split in ["val"]:
                folder_path = os.path.join(self.root_folder, split, "images")
                # make val image folder
                if (
                    sum(
                        [
                            os.path.isdir(os.path.join(self.root_folder, x))
                            for x in os.listdir(folder_path)
                        ]
                    )
                    == 0
                ):
                    print("create val folder")
                    create_val_img_folder(self.root_folder)

            else:
                folder_path = os.path.join(self.root_folder, split)

            labels = sorted(os.listdir(folder_path))

            class_to_idx = {cls: i for i, cls in enumerate(labels)}

            # get image paths
            img_paths, classes = zip(
                *torchvision.datasets.DatasetFolder.make_dataset(
                    folder_path,
                    class_to_idx=class_to_idx,
                    extensions="jpeg",
                )
            )

            if self.coarse_labels:
                classes = sparse2coarse(list(classes))
                self.output_shapes["label"] = (67,)

            split_dp = mk.DataPanel(
                {
                    "image": mk.ImageColumn.from_filepaths(list(img_paths)),
                    "label": mk.TensorColumn(classes),
                }
            )

            dp[split] = split_dp

        self.dataset_train = MeerkatDataset(
            dp["train"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
        self.dataset_val = MeerkatDataset(
            dp["val"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
        self.dataset_test = MeerkatDataset(
            dp["val"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
