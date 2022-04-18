from unagi.data.transforms.image.auto_contrast import AutoContrast
from unagi.data.transforms.image.blur import Blur
from unagi.data.transforms.image.brightness import Brightness
from unagi.data.transforms.image.center_crop import CenterCrop
from unagi.data.transforms.image.color import Color
from unagi.data.transforms.image.color_distortion import ColorDistortion
from unagi.data.transforms.image.color_jitter import ColorJitter
from unagi.data.transforms.image.contrast import Contrast
from unagi.data.transforms.image.cutout import Cutout
from unagi.data.transforms.image.equalize import Equalize
from unagi.data.transforms.image.gaussian_blur import GaussianBlur
from unagi.data.transforms.image.horizontal_filp import HorizontalFlip
from unagi.data.transforms.image.identity import Identity
from unagi.data.transforms.image.invert import Invert
from unagi.data.transforms.image.normalize import Normalize
from unagi.data.transforms.image.posterize import Posterize
from unagi.data.transforms.image.random_crop import RandomCrop
from unagi.data.transforms.image.random_grayscale import RandomGrayscale
from unagi.data.transforms.image.random_horizontal_flip import RandomHorizontalFlip
from unagi.data.transforms.image.random_resize_crop import RandomResizedCrop
from unagi.data.transforms.image.reshape2d import Reshape2D
from unagi.data.transforms.image.resize import Resize
from unagi.data.transforms.image.resize_and_pad import ResizeAndPad
from unagi.data.transforms.image.rotate import Rotate
from unagi.data.transforms.image.sharpness import Sharpness
from unagi.data.transforms.image.shear_x import ShearX
from unagi.data.transforms.image.shear_y import ShearY
from unagi.data.transforms.image.smooth import Smooth
from unagi.data.transforms.image.solarize import Solarize
from unagi.data.transforms.image.to_tensor import ToTensor
from unagi.data.transforms.image.translate_x import TranslateX
from unagi.data.transforms.image.translate_y import TranslateY
from unagi.data.transforms.image.vertical_flip import VerticalFlip

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Blur": Blur,
    "Brightness": Brightness,
    "GaussianBlur": GaussianBlur,
    "CenterCrop": CenterCrop,
    "Color": Color,
    "Contrast": Contrast,
    "Cutout": Cutout,
    "Equalize": Equalize,
    "GaussianBlur": GaussianBlur,
    "ColorDistortion": ColorDistortion,
    "HorizontalFlip": HorizontalFlip,
    "Identity": Identity,
    "Invert": Invert,
    "Posterize": Posterize,
    "RandomCrop": RandomCrop,
    "RandomResizedCrop": RandomResizedCrop,
    "Resize": Resize,
    "Rotate": Rotate,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Smooth": Smooth,
    "Solarize": Solarize,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "VerticalFlip": VerticalFlip,
    "ToTensor": ToTensor,
    "Normalize": Normalize,
    "Reshape2D": Reshape2D,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "ResizeAndPad": ResizeAndPad,
    "ColorJitter": ColorJitter,
    "RandomGrayscale": RandomGrayscale,
}
