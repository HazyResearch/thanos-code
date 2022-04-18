from typing import Any, Dict, List, Tuple, Union

from einops import rearrange
from torch import Tensor

from unagi.trainer.data import default_unagi_collate_fn


def unagi_collate_fn(
    # is_train,
    # feature_type_map,
    # feature_view_map,
    batch: Union[List[Tuple[Dict[str, Any], Dict[str, Tensor]]], List[Dict[str, Any]]],
):
    (x_dict, y_dict) = default_unagi_collate_fn(batch)
    # x_dict["is_train"] = is_train
    # x_dict["feature_type_map"] = feature_type_map
    # x_dict["labels"] = y_dict["labels"]
    """x_dict.update(
        y_dict
    )  # ADD THIS LINE, AND IN YOUR DATALOADER ADD MORE LABELES
    """
    new_x_dict = {}
    new_x_dict["index"] = x_dict["index"]
    del x_dict["index"]
    new_x_dict["inputs"] = x_dict
    new_y_dict = {k: rearrange(v, "b v ... -> (b v) ...") for k, v in y_dict.items()}
    return (new_x_dict, new_y_dict)
