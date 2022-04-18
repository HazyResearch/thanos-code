import random

from PIL import Image

from unagi.data.transforms.image.transform import UnagiTransform
from unagi.data.transforms.image.utils import categorize_value


class TranslateY(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0, max_degree=10):
        self.max_degree = max_degree
        self.value_range = (0, self.max_degree)
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        if random.random() > 0.5:
            degree = -degree
        return (
            pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, degree)),
            label,
        )

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"max_degree={self.max_degree}>"
        )
