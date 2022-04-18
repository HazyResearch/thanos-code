from unagi.data.augmentations.brightness import Brightness
from unagi.data.augmentations.cutout import Cutout
from unagi.data.augmentations.mixup import Mixup

AUGMENTATIONS = {
    "mixup": Mixup,
    # "invert": Invert,
    "cutout": Cutout,
    # "solarize": Solarize,
    "brightness": Brightness,
    # "rotate": Rotate,
}
