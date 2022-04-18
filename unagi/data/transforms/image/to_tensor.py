from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class ToTensor(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0):
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return transforms.ToTensor()(pil_img), label
