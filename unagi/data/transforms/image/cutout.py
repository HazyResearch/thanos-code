import numpy as np
from PIL import ImageDraw

from unagi.data.transforms.image.transform import UnagiTransform
from unagi.data.transforms.image.utils import categorize_value


class Cutout(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0, max_pixel=20, color=None):
        self.max_pixel = max_pixel
        self.value_range = (0, self.max_pixel)
        self.color = color
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        pil_img = pil_img.copy()
        degree = categorize_value(self.level, self.value_range, "int")
        width, height = pil_img.size

        x0 = np.random.uniform(width)
        y0 = np.random.uniform(height)

        x0 = int(max(0, x0 - degree / 2.0))
        y0 = int(max(0, y0 - degree / 2.0))
        x1 = min(width, x0 + degree)
        y1 = min(height, y0 + degree)

        xy = (x0, y0, x1, y1)

        if self.color is not None:
            color = self.color
        elif pil_img.mode == "RGB":
            color = (125, 123, 114)
        elif pil_img.mode == "L":
            color = 121
        else:
            raise ValueError(f"Unspported image mode {pil_img.mode}")

        ImageDraw.Draw(pil_img).rectangle(xy, color)

        return pil_img, label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"max_pixel={self.max_pixel}>"
        )
