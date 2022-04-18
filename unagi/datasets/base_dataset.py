import copy
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from einops import rearrange

from unagi.data.data_utils.transform_util import get_transforms
from unagi.utils.misc import list_to_tensor


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


class TBPTTDataLoader(torch.utils.data.DataLoader):
    """
    Adapted from https://github.com/deepsound-project/samplernn-pytorch
    """

    def __init__(self, dataset, batch_size, chunk_len, overlap_len, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)

        # Zero padding value, given by the dataset
        self.zero = dataset.zero if hasattr(dataset, "zero") else 0

        # Size of the chunks to be fed into the model
        self.chunk_len = chunk_len

        # Keep `overlap_len` from the previous chunk (e.g. SampleRNN requires this)
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            x, y, *z = batch

            # Pad with self.overlap_len - 1 zeros
            x = torch.cat(
                [
                    torch.zeros((x.shape[0], self.overlap_len - 1, *x.shape[2:]))
                    .to(x.device)
                    .to(x.dtype)
                    + self.zero,
                    x,
                ],
                dim=1,
            )
            y = torch.cat(
                [
                    torch.zeros((y.shape[0], self.overlap_len - 1, *y.shape[2:]))
                    .to(y.device)
                    .to(y.dtype)
                    + self.zero,
                    y,
                ],
                dim=1,
            )
            z = [
                torch.cat(
                    [
                        torch.zeros(
                            (
                                z[i].shape[0],
                                self.overlap_len - 1,
                                *z[i].shape[2:],
                            )
                        )
                        .to(z[i].device)
                        .to(z[i].dtype),
                        z[i],
                    ],
                    dim=1,
                )
                for i in range(len(z))
            ]

            _, seq_len, *_ = x.shape

            reset = True

            for seq_begin in list(range(self.overlap_len - 1, seq_len, self.chunk_len))[
                :-1
            ]:
                from_index = seq_begin - self.overlap_len + 1
                to_index = seq_begin + self.chunk_len
                # TODO: check this
                # Ensure divisible by overlap_len
                if self.overlap_len > 0:
                    to_index = min(
                        to_index,
                        seq_len - ((seq_len - self.overlap_len + 1) % self.overlap_len),
                    )

                x_chunk = x[:, from_index:to_index]
                if len(y.shape) == 3:
                    y_chunk = y[:, seq_begin:to_index]
                else:
                    y_chunk = y
                z_chunk = [z_[:, from_index:to_index] for z_ in z]

                yield (x_chunk, y_chunk, *z_chunk, reset)

                reset = False

    def __len__(self):
        raise NotImplementedError()


class UnagiDatasetBuilder:
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name.")
    # Important:
    # - dataset should return x, y in the order in which the keys
    #    are defined in the dicts below
    # - the shapes should always have channels first
    input_shapes: dict = NotImplementedError("Dataset must have input shapes.")
    output_shapes: dict = NotImplementedError("Dataset must have output shapes.")
    uid = "index"

    @property
    def x_names(self):
        self._x_names = list(self.input_shapes.keys())
        return self._x_names

    @property
    def y_names(self):
        self._y_names = list(self.output_shapes.keys())
        return self._y_names

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    @property
    def init_defaults(self):
        return {}

    def __init__(
        self,
        data_dir=None,
        tbptt=False,
        chunk_len=None,
        overlap_len=None,
        x_transforms=None,
        y_transforms=None,
        transform_pool=None,
        **dataset_cfg,
    ):
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.transform_pool = transform_pool

        # Arguments for TBPTT: only used if tbptt is True and are passed to
        #  TBPTTDataLoader
        self.tbptt = tbptt
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len
        self.subset_split_percent = None
        self.subset_split_seed = 42

        # Add all arguments to self
        init_args = self.init_defaults
        init_args.update(
            dataset_cfg
        )  # TODO this overrides the default dict which is bad
        for k, v in init_args.items():
            setattr(self, k, v)

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.setup()
        self.subsample_dataset()
        self._wrap_datasets()

    def setup(self):
        """
        This method should set
        self.dataset_train, self.dataset_val, and self.dataset_test.
        """
        raise NotImplementedError

    def subsample_dataset(self):
        if self.subset_split_percent:
            subset_size = int(len(self.dataset_train) * self.subset_split_percent)
            dataset_train_subset, dataset_train_heldout = torch.utils.data.random_split(
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

    def _wrap_datasets(self):
        self._create_input_to_output_mapping()

        # Create all the transforms
        self.transforms_train = self.transforms("train")
        self.transforms_eval = self.transforms("eval")

        UnagiDatasetWrapped = partial(
            UnagiDataset,
            input_to_output_mapping=self.input_to_output_mapping,
            x_names=self.x_names,
            y_names=self.y_names,
            uid=self.uid,
        )

        if isinstance(self.dataset_train, torch.utils.data.Dataset):
            self.dataset_train = UnagiDatasetWrapped(
                dataset=self.dataset_train,
                feature_transforms=self.transforms_train,
                split="train",
            )
        elif isinstance(self.dataset_train, dict):
            self.dataset_train = {
                k: UnagiDatasetWrapped(
                    dataset=v,
                    feature_transforms=self.transforms_train,
                    split=k,
                )
                for k, v in self.dataset_train.items()
            }
        else:
            raise TypeError(
                "dataset_train must be a torch.utils.data.Dataset or dict,"
                f"got {type(self.dataset_train)}"
            )

        if isinstance(self.dataset_val, torch.utils.data.Dataset):
            self.dataset_val = UnagiDatasetWrapped(
                dataset=self.dataset_val,
                feature_transforms=self.transforms_eval,
                split="val",
            )
        elif isinstance(self.dataset_val, dict):
            self.dataset_val = {
                k: UnagiDatasetWrapped(
                    dataset=v,
                    feature_transforms=self.transforms_eval,
                    split=k,
                )
                for k, v in self.dataset_val.items()
            }
        else:
            raise TypeError(
                "dataset_val must be a torch.utils.data.Dataset or dict, "
                f"got {type(self.dataset_val)}"
            )

        if isinstance(self.dataset_test, torch.utils.data.Dataset):
            self.dataset_test = UnagiDatasetWrapped(
                dataset=self.dataset_test,
                feature_transforms=self.transforms_eval,
                split="test",
            )
        elif isinstance(self.dataset_test, dict):
            self.dataset_test = {
                k: UnagiDatasetWrapped(
                    dataset=v,
                    feature_transforms=self.transforms_eval,
                    split=k,
                )
                for k, v in self.dataset_test.items()
            }
        else:
            raise TypeError(
                "dataset_test must be a torch.utils.data.Dataset or dict, "
                f"got {type(self.dataset_test)}"
            )

    def split_train_val(self, val_split: float):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        dataset_train, dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
        )
        return dataset_train, dataset_val

    def transforms(self, split: str):
        # returns a Composed transform
        return get_transforms(
            input_features=self.x_transforms,
            dataset_split=split,
            augmentations=self.transform_pool,
        )

    def _create_input_to_output_mapping(self):
        # for contrastive, keep track of which input features as are transformed
        # with which output features
        self.input_to_output_mapping = {}
        for name, output_feat in self.y_transforms.items():
            if "transform_with" in output_feat:
                input_feature_map = output_feat.transform_with
                if input_feature_map not in self.input_to_output_mapping:
                    self.input_to_output_mapping[input_feature_map] = [name]
                else:
                    self.input_to_output_mapping[input_feature_map].append(name)

    @staticmethod
    def collate_fn(
        batch: Union[
            List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]], List[Dict[str, Any]]
        ],
        resolution: int = 1,
        is_train: bool = True,
    ) -> Union[Tuple[Dict[str, Any], Dict[str, torch.Tensor]], Dict[str, Any]]:
        """Collate function.

        Args:
            batch: The batch to collate.
            min_data_len: The minimal data sequence length, defaults to 0.
            max_data_len: The maximal data sequence length (0 means no limit),
                        defaults to 0.
        Returns:
            The collated batch.
        """
        X_batch: defaultdict = defaultdict(list)
        Y_batch: defaultdict = defaultdict(list)

        for item in batch:
            # Check if batch is (x_dict, y_dict) pair
            if isinstance(item, dict):
                x_dict = item
                y_dict: Dict[str, Any] = dict()
            else:
                x_dict, y_dict = item
            for field_name, value in x_dict.items():
                if isinstance(value, list):
                    X_batch[field_name] += value
                else:
                    X_batch[field_name].append(value)
            for label_name, value in y_dict.items():
                if isinstance(value, list):
                    Y_batch[label_name] += value
                else:
                    Y_batch[label_name].append(value)

        field_names = copy.deepcopy(list(X_batch.keys()))

        for field_name in field_names:
            values = X_batch[field_name]
            # Only merge list of tensors
            if isinstance(values[0], torch.Tensor):
                item_tensor, item_mask_tensor = list_to_tensor(
                    values,
                )
                X_batch[field_name] = item_tensor
                if item_mask_tensor is not None:
                    X_batch[f"{field_name}_mask"] = item_mask_tensor

                # TODO: generalize this to handle the case where resolutions are
                # different per field
                X_batch[field_name] = X_batch[field_name][..., ::resolution]
                # TODO: figure out how to handle the mask
                # X_batch[f"{field_name}_mask"] = X_batch[f"{field_name}_mask"]
                #                                           [..., ::resolution]

        for label_name, values in Y_batch.items():
            Y_batch[label_name] = list_to_tensor(
                values,
            )[0]

        X_batch = dict(X_batch)
        X_batch["is_train"] = is_train
        new_X_batch = {}
        new_X_batch["index"] = X_batch["index"]
        del X_batch["index"]
        new_X_batch["inputs"] = X_batch

        if len(Y_batch) != 0:
            Y_batch = dict(Y_batch)
            Y_batch = {
                k: rearrange(v, "b v ... -> (b v) ...") for k, v in Y_batch.items()
            }
            return new_X_batch, Y_batch

        return new_X_batch

    def train_dataloader(self, train_resolution=None, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now"

        return self._dataloader(
            self.dataset_train,
            is_train=True,
            resolutions=train_resolution,
            shuffle=True,
            **kwargs,
        )[0]

    def val_dataloader(self, eval_resolutions=None, **kwargs):
        if isinstance(self.dataset_val, dict):
            val_dataloaders = {}
            for prefix, dataset in self.dataset_val.items():
                dls = self._eval_dataloader(
                    dataset,
                    prefix=f"val/{prefix}",
                    eval_resolutions=eval_resolutions,
                    **kwargs,
                )
                val_dataloaders = {**val_dataloaders, **dls}
            return val_dataloaders
        else:
            return self._eval_dataloader(
                self.dataset_val,
                prefix="val",
                eval_resolutions=eval_resolutions,
                **kwargs,
            )

    def test_dataloader(self, eval_resolutions=None, **kwargs):
        if isinstance(self.dataset_test, dict):
            test_dataloaders = {}
            for prefix, dataset in self.dataset_test.items():
                dls = self._eval_dataloader(
                    dataset,
                    prefix=f"test/{prefix}",
                    eval_resolutions=eval_resolutions,
                    **kwargs,
                )
                test_dataloaders = {**test_dataloaders, **dls}
            return test_dataloaders
        else:
            return self._eval_dataloader(
                self.dataset_test,
                prefix="test",
                eval_resolutions=eval_resolutions,
                **kwargs,
            )

    def _eval_dataloader(self, dataset, prefix, eval_resolutions=None, **kwargs):
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]

        kwargs["shuffle"] = False if "shuffle" not in kwargs else kwargs["shuffle"]
        dataloaders = self._dataloader(
            dataset,
            is_train=False,
            resolutions=eval_resolutions,
            **kwargs,
        )

        return (
            {
                f"{prefix}/{res}" if res > 1 else prefix: dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None
            else None
        )

    def _dataloader(self, dataset, is_train, resolutions, **loader_args):
        if dataset is None:
            return None

        if self.tbptt:
            DataLoader = partial(
                TBPTTDataLoader,
                chunk_len=self.chunk_len,
                overlap_len=self.overlap_len,
            )
        else:
            DataLoader = torch.utils.data.DataLoader

        return [
            DataLoader(
                dataset=dataset,
                collate_fn=partial(
                    self.collate_fn, resolution=resolution, is_train=is_train
                )
                if self.collate_fn is not None
                else None,
                **loader_args,
            )
            for resolution in resolutions
        ]

    def __str__(self):
        return self._name_


class UnagiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        feature_transforms: dict,
        input_to_output_mapping: dict,
        x_names: list,
        y_names: list,
        uid: str,
        split: str,
    ):
        self.dataset = dataset
        self.feature_transforms = feature_transforms
        self.input_to_output_mapping = input_to_output_mapping
        self.x_names = x_names
        self.y_names = y_names
        self.uid = uid
        self.split = split

    def __getitem__(self, index):
        """
        Get item by index.

        Args:
          index(index): The index of the item.
        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """

        x, y = self.dataset[index]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        x_dict, y_dict = {}, {}
        # add uid
        x_dict[self.uid] = index
        for idx, input_feature in enumerate(self.x_names):
            if len(self.x_names) > 1:
                feature = x[idx]
            else:
                feature = x
            x_dict[input_feature] = []
            transforms = self.feature_transforms[input_feature]
            if input_feature in self.input_to_output_mapping:
                output_targets = self.input_to_output_mapping[input_feature]
                labels = torch.stack(
                    [
                        y[self.y_names.index(target)] if len(self.y_names) > 1 else y
                        for target in output_targets
                    ]
                )
                feature, y = transforms(
                    feature,  # input
                    labels,  # label
                )
            else:
                feature, _ = transforms(
                    feature,  # input
                    None,  # label
                )
            x_dict[input_feature].append(feature)
        for index, output_feature in enumerate(self.y_names):
            y_dict[output_feature] = y[index]
        return x_dict, y_dict

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = UnagiDataset(
        data_dir=".",
        tbptt=True,
        chunk_len=10,
        overlap_len=5,
        permute=False,
        n_classes=20,
    )
