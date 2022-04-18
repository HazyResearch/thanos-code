from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class RandomResizedCrop(UnagiTransform):
    def __init__(
        self,
        size,
        name=None,
        prob=1.0,
        level=0,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.333_333_333_333_333_3),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.transform_func = transforms.RandomResizedCrop(
            self.size, self.scale, self.ratio, self.interpolation
        )

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"size={self.size}, scale={self.scale}, ratio={self.ratio}, "
            f"interpolation={self.interpolation}>"
        )
