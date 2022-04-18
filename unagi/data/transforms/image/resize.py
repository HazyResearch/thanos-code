from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class Resize(UnagiTransform):
    def __init__(
        self,
        size,
        name=None,
        prob=1.0,
        level=0,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.size = size
        self.interpolation = interpolation
        self.transform_func = transforms.Resize(self.size, self.interpolation)

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}, interpolation={self.interpolation}>"
        )
