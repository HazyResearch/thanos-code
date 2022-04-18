from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class ColorJitter(UnagiTransform):
    def __init__(
        self,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        name=None,
        prob=1.0,
        level=0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform_func = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f" brightness={self.brightness}, contrast={self.contrast}, "
            f" saturation={self.saturation}, hue={self.hue}"
        )
