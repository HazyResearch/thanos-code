from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class RandomHorizontalFlip(UnagiTransform):
    def __init__(self, p=0.5, name=None, prob=1.0, level=0):
        self.p = p
        self.transform_func = transforms.RandomHorizontalFlip(p)

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"p={self.p}"
        )
