import torch
from torch import nn
from torchvision import transforms as transforms


class Grayscale(nn.Module):
    def __init__(self, dim=1, resize=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.resize = resize
        if self.resize:
            self.resize_func = transforms.Resize(
                self.resize, transforms.InterpolationMode.BILINEAR
            )

    def forward(self, x):
        grayscale_image = torch.mean(x, dim=self.dim, keepdim=True)
        if self.resize:
            return self.resize_func(grayscale_image)

        return grayscale_image
