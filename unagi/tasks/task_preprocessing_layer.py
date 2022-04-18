import copy

import torch
import torch.nn as nn
from einops import rearrange


class MaskInputTransforms(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "masked_input_transform"

    def forward(self, x_batch):
        temp_x_batch = copy.deepcopy(x_batch)
        is_train = temp_x_batch["is_train"]

        field_names = copy.deepcopy(list(temp_x_batch.keys()))
        new_x_batch = {"inputs": {}, "mask": {}}

        for field_name in field_names:
            if field_name not in [
                "is_train",
            ]:
                values = temp_x_batch[field_name]
                input = tuple(map(torch.stack, zip(*values)))
                inputs, mask = input
                if is_train:
                    mask = rearrange(mask, "b v d -> (b v) d")
                else:
                    mask = None

                new_x_batch["inputs"][field_name] = {
                    "value": inputs,
                }
                new_x_batch["mask"][field_name] = mask

        return (
            new_x_batch["inputs"],
            new_x_batch["inputs"],
            new_x_batch["mask"],
            temp_x_batch["labels"],
        )  # (src, tgt, mask)


class ContrastiveInputTransforms(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "contrastive_input_transform"

    def forward(self, x_batch):
        """temp_x_batch = copy.deepcopy(x_batch)
        is_train = temp_x_batch["is_train"]
        feature_type_map = temp_x_batch["feature_type_map"]

        field_names = copy.deepcopy(list(temp_x_batch.keys()))
        new_x_batch = {"inputs": {}}

        for field_name in field_names:
            if field_name not in ["is_train", "feature_type_map", "labels"]:
                values = temp_x_batch[field_name]
                if is_train:
                    values = rearrange(values, "b v c d -> (b v) c d")

                new_x_batch["inputs"][field_name] = {
                    "value": values,
                    "type": feature_type_map[field_name],
                }

        return [new_x_batch["inputs"], temp_x_batch["labels"]]"""
        temp_x_batch = copy.deepcopy(x_batch)
        for param, values in temp_x_batch.items():
            if param not in [
                "is_train",
            ]:
                values = temp_x_batch[param]
                values = rearrange(values, "b v ... -> (b v) ...")
                temp_x_batch[param] = values
        del temp_x_batch["is_train"]

        return temp_x_batch


class CLIPInputTransforms(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "clip_input_transform"

    def forward(self, x_batch):
        """temp_x_batch = copy.deepcopy(x_batch)
        is_train = temp_x_batch["is_train"]
        feature_type_map = temp_x_batch["feature_type_map"]

        field_names = copy.deepcopy(list(temp_x_batch.keys()))

        final_output = {}
        for field_name in field_names:
            new_x_batch = {field_name: {}}
            if field_name not in ["is_train", "feature_type_map", "labels"]:
                values = temp_x_batch[field_name]
                if is_train:
                    values = rearrange(values, "b v c d -> (b v) c d")

                new_x_batch[field_name] = {
                    "value": values,
                    "type": feature_type_map[field_name],
                }

                if feature_type_map[field_name] == "text":
                    final_output["text"] = new_x_batch
                elif feature_type_map[field_name] == "image":
                    final_output["image"] = new_x_batch

        final_output["labels"] = temp_x_batch["labels"]"""
        temp_x_batch = copy.deepcopy(x_batch)
        for param, values in temp_x_batch.items():
            if param not in [
                "is_train",
            ]:
                values = temp_x_batch[param]
                values = rearrange(values, "b v ... -> (b v) ...")
                temp_x_batch[param] = values
        del temp_x_batch["is_train"]

        return temp_x_batch


class SupervisedInputTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "supervised_input_transform"

    def forward(self, x_batch):
        # is_train = False
        """temp_x_batch = copy.deepcopy(x_batch)
        feature_type_map = temp_x_batch["feature_type_map"]

        field_names = copy.deepcopy(list(temp_x_batch.keys()))
        new_x_batch = {"inputs": {}}
        for field_name in field_names:
            if field_name not in ["is_train", "feature_type_map", "labels"]:
                values = temp_x_batch[field_name]
                if field_name in feature_type_map.keys():
                    new_x_batch["inputs"][field_name] = {
                        "value": values,
                        "type": feature_type_map[field_name],
                    }
            if field_name == "is_train":
                is_train = temp_x_batch[field_name]"""

        # pass foward inputs and labels
        # return [new_x_batch["inputs"], temp_x_batch["labels"], is_train]
        temp_x_batch = copy.deepcopy(x_batch)
        for param, values in temp_x_batch.items():
            if param not in [
                "is_train",
            ]:
                values = temp_x_batch[param]
                values = rearrange(values, "b v ... -> (b v) ...")
                temp_x_batch[param] = values
        del temp_x_batch["is_train"]

        return temp_x_batch


class ViewSelect(nn.Module):
    def __init__(self, view_idx, n_views, **kwargs):
        super().__init__()
        self.name = "view_select"
        self.view_idx = view_idx
        self.n_views = n_views

    def forward(self, input):
        embs = rearrange(input, "(b v) ... -> b v ...", v=self.n_views)
        embs = embs[:, self.view_idx, ...]
        return embs
