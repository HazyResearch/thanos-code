from unagi.data.transforms.image import ALL_TRANSFORMS as ALL_IMAGE_TRANSFORMS
from unagi.data.transforms.task import ALL_TRANSFORMS as ALL_TASK_TRANSFORMS
from unagi.data.transforms.text import ALL_TRANSFORMS as ALL_TEXT_TRANSFORMS

ALL_TRANSFORMS = {
    "text": ALL_TEXT_TRANSFORMS,
    "image": ALL_IMAGE_TRANSFORMS,
    "task": ALL_TASK_TRANSFORMS,
}
