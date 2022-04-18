from PIL import Image, ImageOps
from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class ResizeAndPad(UnagiTransform):
    def __init__(
        self,
        resized_width,
        resized_height,
        name=None,
        prob=1.0,
        level=0,
        ratio=(0.75, 1.333_333_333_333_333_3),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.resized_height = resized_height
        self.resized_width = resized_width

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        original_size = pil_img.size
        ratio = float(self.resized_width) / max(original_size)
        new_size = (int(self.resized_width * ratio), int(self.resized_height * ratio))
        pil_img = pil_img.resize(new_size, Image.ANTIALIAS)
        delta_w = self.resized_width - new_size[0]
        delta_h = self.resized_height - new_size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        resized_img = ImageOps.expand(pil_img, padding)
        return resized_img

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}, scale={self.scale}, ratio={self.ratio}, "
            f"interpolation={self.interpolation}>"
        )
