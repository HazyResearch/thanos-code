import os

import meerkat as mk
import numpy as np
import pandas as pd
import torch

from unagi.datasets.base_dataset import UnagiDatasetBuilder
from unagi.datasets.meerkat_dataset import MeerkatDataset


class CelebA(UnagiDatasetBuilder):
    """Dataset to load CelebA dataset."""

    _name_ = "celeba"
    # TODO: these can be modified by the transforms (e.g. grayscale)
    # and need to be up to date
    input_shapes = {
        "image": (3, 224, 224),
    }
    output_shapes = {
        "label": (10,),
    }

    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,  # For validation split
            "target_name": "Blond_Hair",
            "confounder_names": ["Male"],
            "root_dir": None,
        }

    def setup(self):
        if not os.path.exists(self.root_dir):
            raise ValueError(
                f"{self.root_dir} does not exist yet. "
                f"Please generate the dataset first."
            )

        self.metadata_df = pd.read_csv(
            os.path.join(self.root_dir, "list_attr_celeba.csv"),
            delim_whitespace=True,
        )
        self.split_df = pd.read_csv(
            os.path.join(self.root_dir, "list_eval_partition.csv"),
            delim_whitespace=True,
        )

        dp = {"train": [], "test": [], "val": []}

        for split in ["train", "test", "val"]:

            self.metadata_df["partition"] = self.split_df["partition"]
            self.metadata_df = self.metadata_df[
                self.split_df["partition"] == self.split_dict[split]
            ]

            self.y_array = self.metadata_df[self.target_name].values

            self.confounder_array = self.metadata_df[self.confounder_names].values
            self.y_array[self.y_array == -1] = 0
            self.confounder_array[self.confounder_array == -1] = 0
            self.n_classes = len(np.unique(self.y_array))
            self.n_confounders = len(self.confounder_names)

            self.output_shapes = (self.n_classes,)

            # Get sub_targets / group_idx
            self.metadata_df["sub_target"] = (
                self.metadata_df[self.target_name].astype(str)
                + "_"
                + self.metadata_df[self.confounder_names].astype(str)
            )
            # print('> Sub_target loaded!')

            # Get subclass map
            attributes = [self.target_name, self.confounder_names]
            self.df_groups = (
                self.metadata_df[attributes].groupby(attributes).size().reset_index()
            )
            # print('> Groups loaded!')
            self.df_groups["group_id"] = (
                self.df_groups[self.target_name].astype(str)
                + "_"
                + self.df_groups[self.confounder_names].astype(str)
            )
            # print('> Group IDs loaded!')
            self.subclass_map = (
                self.df_groups["group_id"]
                .reset_index()
                .set_index("group_id")
                .to_dict()["index"]
            )
            self.group_array = (
                self.metadata_df["sub_target"].map(self.subclass_map).values
            )
            groups, group_counts = np.unique(self.group_array, return_counts=True)
            self.n_groups = len(groups)

            # Extract filenames and splits
            self.filename_array = self.metadata_df["image_id"].values
            self.split_array = self.metadata_df["partition"].values

            self.targets = torch.tensor(self.y_array)
            self.targets_all = {
                "target": np.array(self.y_array),
                "group_idx": np.array(self.group_array),
                "spurious": np.array(self.confounder_array),
                "sub_target": np.array(list(zip(self.y_array, self.confounder_array))),
            }
            self.group_labels = [self.group_str(i) for i in range(self.n_groups)]

            file_paths = [
                os.path.join(self.root_dir, "img_align_celeba", fname)
                for fname in self.filename_array
            ]

            dp[split] = mk.DataPanel(
                {
                    "image": mk.ImageColumn.from_filepaths(file_paths),
                    # "label": mk.TensorColumn(label),
                    "label": mk.TensorColumn(self.targets),
                }
            )

        self.dataset_train = MeerkatDataset(
            dp["train"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
        self.dataset_test = MeerkatDataset(
            dp["test"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
        self.dataset_test = MeerkatDataset(
            dp["val"], xs=list(self.input_shapes.keys()), ys=["label"]
        )
