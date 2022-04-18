from torchvision import transforms

from unagi.data.transforms.image.transform import UnagiTransform


class ColorDistortion(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0, strength=0.5):
        super().__init__(name, prob, level)
        self.strength = strength
        self.color_jitter = transforms.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)
        self.color_distort = transforms.Compose([self.rnd_color_jitter, self.rnd_gray])

    def transform(self, pil_img, label, **kwargs):
        return self.color_distort(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), prob={self.prob}, level={self.level}, "
            f"strength={self.strength}"
        )
