""" Unagi DataLoader and default collate fn
    code inspiration: emmental

"""
import copy
import logging
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.utils.data import DataLoader

from unagi.datasets.base_dataset import UnagiDataset
from unagi.utils.misc import list_to_tensor

logger = logging.getLogger(__name__)


def default_unagi_collate_fn(
    batch: Union[List[Tuple[Dict[str, Any], Dict[str, Tensor]]], List[Dict[str, Any]]],
) -> Union[Tuple[Dict[str, Any], Dict[str, Tensor]], Dict[str, Any]]:
    """Collate function.
    Args:
      batch: The batch to collate.
      min_data_len: The minimal data sequence length, defaults to 0.
      max_data_len: The maximal data sequence length (0 means no limit), defaults to 0.
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
        if isinstance(values[0], Tensor):
            item_tensor, item_mask_tensor = list_to_tensor(
                values,
            )
            X_batch[field_name] = item_tensor
            if item_mask_tensor is not None:
                X_batch[f"{field_name}_mask"] = item_mask_tensor
    for label_name, values in Y_batch.items():
        Y_batch[label_name] = list_to_tensor(
            values,
        )[0]

    if len(Y_batch) != 0:
        return dict(X_batch), dict(Y_batch)
    else:
        return dict(X_batch)


# class UnagiDataLoader(DataLoader):
#     """UnagiDataLoader
#     An advanced dataloader class which contains mapping from task to label (which
#     label(s) to use in dataset's Y_dict for this task), and split (which part this
#     dataset belongs to) information.
#     Args:
#       task_to_label_dict: The task to label mapping where key is the task name
#         and value is the label(s) for that task and should be the key in Y_dict.
#       dataset: The dataset to construct the dataloader
#       split: The split information, defaults to "train".
#       collate_fn: The function that merges a list of samples to
#         form a mini-batch, defaults to emmental_collate_fn.
#       n_batches: Total number of batches.
#       **Kwargs: Other arguments of dataloader.
#     """

#     def __init__(
#         self,
#         dataset: UnagiDataset,
#         split: str = "train",
#         collate_fn: Optional[Callable] = None,
#         # n_batches: Optional[int] = None,
#         **kwargs: Any,
#     ) -> None:
#         """Initialize UnagiDataLoader."""
#         assert isinstance(
#             dataset, UnagiDataset
#         ), "dataset should inherent from UnagiDataset."

#         if collate_fn is None:
#             collate_fn = partial(default_unagi_collate_fn)

#         super().__init__(dataset, collate_fn=collate_fn, **kwargs)

#         # self.data_name = dataset._name_
#         # self.uid = dataset.uid
#         self.split = split
#         # self.n_batches = n_batches
