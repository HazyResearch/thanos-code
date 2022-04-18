from torchvision import transforms

from unagi.data.transforms.image.transform import UnagiTransform


class GaussianBlur(UnagiTransform):
    def __init__(self, kernel_size, sigma=(0.1, 2.0), name=None, prob=1.0, level=0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.transform_func = transforms.GaussianBlur(self.kernel_size, self.sigma)

        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label
