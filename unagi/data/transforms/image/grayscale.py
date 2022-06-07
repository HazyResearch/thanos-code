from torchvision import transforms as transforms

from unagi.data.transforms.image.transform import UnagiTransform


class Grayscale(UnagiTransform):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels
        self.transform_func = transforms.Grayscale(self.num_output_channels)

        super().__init__(name="Grayscale", prob=1.0, level=0)

    def transform(self, pil_img, label, **kwargs):
        return self.transform_func(pil_img), label

    def __repr__(self):
        return (
            f"<Transform ({self.name}), num_output_channels={self.num_output_channels}>"
        )
