from unagi.data.transforms.task.transform import (
    GroupTransform,
    IdentityTransform,
    MaskGen,
    TupleTransform,
)

ALL_TRANSFORMS = {
    "Contrastive": GroupTransform,
    "MaskGenerator": MaskGen,
    "Mask": TupleTransform,
    "Identity": IdentityTransform,
}
