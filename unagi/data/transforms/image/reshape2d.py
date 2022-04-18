from unagi.data.transforms.image.transform import UnagiTransform


class Reshape2D(UnagiTransform):
    def __init__(self, h_dim, w_dim, name=None, prob=1.0, level=0):
        self.h_dim = h_dim
        self.w_dim = w_dim
        super().__init__(name, prob, level)

    def transform(self, pil_img, label, **kwargs):
        return pil_img.view(self.h_dim, self.w_dim), label
