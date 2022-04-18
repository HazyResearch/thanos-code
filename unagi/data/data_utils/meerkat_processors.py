# flake8: noqa
# from __future__ import annotation

import logging
from typing import Collection, List, Sequence, Tuple

import meerkat as mk
import pandas as pd
import torch
from meerkat.columns.lambda_column import LambdaColumn
from meerkat.tools.lazy_loader import LazyLoader

from unagi.data.transforms.task import GroupTransform, TupleTransform

folder = LazyLoader("torchvision.datasets.folder")

logger = logging.getLogger(__name__)

"""
class MultiImageColumn(LambdaColumn):
    def __init__(
        self,
        data: Sequence[Tuple(str, str)] = None,
        transform: List[callable] = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        super(MultiImageColumn, self).__init__(
            mk.PandasSeriesColumn.from_data(data), *args, **kwargs
        )
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform

    def fn(self, filepaths: Tuple(str, str)):
        image_0, image_1 = self.loader(filepaths[0]), self.loader(filepaths[1])
        image_0, image_1 = self.transform[0](image_0), self.transform[1](image_1)
        image_cat = torch.cat((image_0, image_1), 1)
        return self.transform[2](image_cat)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: List[Sequence[str]],
        loader: callable = None,
        transform: List[callable] = None,
        *args,
        **kwargs,
    ):
        return cls(data=filepaths, loader=loader, transform=transform, *args, **kwargs)

    @classmethod
    def default_loader(cls, *args, **kwargs):
        return folder.default_loader(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> Collection:
        return (super()._state_keys() | {"transform", "loader"}) - {"fn"}

    def _repr_pandas_(self) -> pd.Series:
        return "ImageCell(" + self.data.data.reset_index(drop=True) + ")"
"""


class TextTransformCell(mk.AbstractCell):
    def __init__(self, input_text: str, transforms):
        self.input = input_text
        self.transforms = transforms
        self._token_ids = None

    def get(self):
        if self._token_ids is None:
            token_ids = self.transforms(self.input, None)[0]
            self._token_ids = token_ids
        return self._token_ids

    def data(self):
        return self.input

    def __repr__(self):
        return "TextTransformCell"


class PILImgTransformCell(mk.AbstractCell):
    def __init__(self, pil_image, transforms):
        self.pil_image = pil_image
        self.transforms = transforms

    def get(self):
        if self.transforms is None:
            return self.pil_image
        else:
            transformed_img = self.transforms(self.pil_image, None)
            if not isinstance(self.transforms, TupleTransform) and not isinstance(
                self.transforms, GroupTransform
            ):
                transformed_img = transformed_img[0]
            return transformed_img

    def data(self):
        return self.pil_image

    def transforms(self):
        return self.transforms

    def __repr__(self):
        return "PILImgTransformCell"
