from PIL import ImageEnhance

from unagi.data.transforms.image.transform import UnagiTransform
from unagi.data.transforms.image.utils import categorize_value


class Brightness(UnagiTransform):

    value_range = (0.1, 1.9)

    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        degree = categorize_value(self.level, self.value_range, "float")
        return ImageEnhance.Brightness(pil_img).enhance(degree), label
