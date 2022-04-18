import random

from unagi.data.transforms.image.cutout import Cutout as CutoutTransform
from unagi.data.transforms.image.transform import UnagiTransform


class Cutout(UnagiTransform):
    def __init__(
        self,
        name=None,
        prob=1.0,
        level=0,
        alpha=1.0,
        same_class_ratio=-1.0,
        prob_label=False,
    ):
        self.alpha = alpha
        self.same_class_ratio = same_class_ratio
        self.prob_label = prob_label
        self.cutout = CutoutTransform(prob=prob, level=level)

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, dp_x, dp_y):
        if random.random() < self.prob:
            cutout_img, cutout_label = self.cutout(pil_img, label)
            return cutout_img, cutout_label
        else:
            return pil_img, label
