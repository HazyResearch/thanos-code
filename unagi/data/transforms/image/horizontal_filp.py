from PIL import Image

from unagi.data.transforms.image.transform import UnagiTransform


class HorizontalFlip(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT), label
