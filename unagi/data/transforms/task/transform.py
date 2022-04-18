import torch

from unagi.data.transforms.image.compose import Compose


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, x, label):
        return x, label


class GroupTransform:
    def __init__(self, transform, views=2):
        self.t = transform
        self.views = views
        self.squeeze = False

    def __call__(self, x, label, **kwargs):
        grouped_contrastive_transforms_lst, label_lst = zip(
            *[self.t(x, label, **kwargs) for i in range(self.views)]
        )
        grouped_contrastive_transforms = torch.stack(grouped_contrastive_transforms_lst)
        if label is not None:
            label = torch.stack(label_lst, dim=1)

        return grouped_contrastive_transforms, label


class MaskGen:
    def __init__(self, views, mask_length, mask_prob=0.05):
        self.mask_length = mask_length
        self.views = views
        self.mask_prob = mask_prob

    def __call__(self, x, label, **kwargs):
        return torch.rand(self.views, self.mask_length) < self.mask_prob


class TupleTransform:
    def __init__(self, *args):
        self.fs = args

    def __call__(self, x, label, **kwargs):
        input = [
            f(x, label, **kwargs)[0]
            if isinstance(f, Compose)
            else f(x, label, **kwargs)
            for f in self.fs
        ]
        for f in self.fs:
            if isinstance(f, Compose):
                label = f(x, label, **kwargs)[1]
        return input, label
        # return tuple([f(x, label)[0] for f in self.fs])
