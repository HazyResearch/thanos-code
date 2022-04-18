import random
from typing import Tuple

import numpy as np
import torch

from unagi.data.transforms.image.transform import UnagiTransform


class Mixup(UnagiTransform):
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

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, dp_x, dp_y):
        """
        Note: for Mixup apply all transforms (including converting to Tesnsor)
                before applying mixup

        pil_img (Tensor)
        dp_x: input column
        dp_y: label column
        """
        if random.random() < self.prob:
            if self.alpha > 0.0:
                mix_ratio = np.random.beta(self.alpha, self.alpha)
            else:
                mix_ratio = 1.0

            idx = np.random.randint(len(dp_x))
            tot_cnt = len(dp_x)

            if self.same_class_ratio >= 0:  # get idx
                same_class = (
                    True if np.random.rand() <= self.same_class_ratio else False
                )
                for i in np.random.permutation(tot_cnt):
                    if same_class == torch.equal(dp_y["labels"][i], label):
                        idx = i
                        break

            cand_img = dp_x[idx]
            cand_label = dp_y[idx]
            # Calc all transforms before mixup

            if isinstance(cand_img, Tuple):
                cand_img = cand_img[0]
            if isinstance(pil_img, Tuple):
                cand_img = pil_img[0]

            mixup_img = mix_ratio * pil_img + (1 - mix_ratio) * cand_img

            if label is not None:
                if self.prob_label:
                    mixup_label = mix_ratio * label + (1 - mix_ratio) * cand_label
                else:
                    mixup_label = (
                        label if np.random.random() < mix_ratio else cand_label
                    )
            else:
                mixup_label = label

            return mixup_img, mixup_label

        else:

            return pil_img, label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"alpha={self.alpha}, same_class_ratio={self.same_class_ratio}, "
            f"prob_label={self.prob_label}>"
        )
