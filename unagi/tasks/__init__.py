from torch.nn import (
    BCEWithLogitsLoss as BCELoss,
    CrossEntropyLoss as CELoss,
    L1Loss as L1Loss,
    MSELoss as MSELoss,
)

from unagi.tasks.loss_fns.ce_loss import LabelSmoothing, SoftCrossEntropyLoss
from unagi.tasks.loss_fns.contrastive_loss import ContrastiveLoss
from unagi.tasks.loss_fns.mask_loss import BatchMask, BatchMaskDup

"""from unagi.tasks.loss_modules import (
    ce_loss,
    contrastive_loss,
    contrastive_loss_clip,
    mask_loss,
    sce_loss,
)"""
from unagi.tasks.output_layer_modules import (
    multiclass_classification,
    multilabel_classification,
)
from unagi.tasks.task_preprocessing_layer import (
    CLIPInputTransforms,
    ContrastiveInputTransforms,
    MaskInputTransforms,
    SupervisedInputTransform,
    ViewSelect,
)

LOSS_MODULE_REGISTRY = {
    "sup_con": ContrastiveLoss,
    "sim_clr": ContrastiveLoss,
    "l_spread": ContrastiveLoss,
    "l_attract": ContrastiveLoss,
    "batch_mask": BatchMask,
    "batch_mask_dup": BatchMaskDup,
    "label_smoothing": LabelSmoothing,
    "cross_entropy": CELoss,
    "binary_cross_entropy": BCELoss,
    "soft_cross_entropy": SoftCrossEntropyLoss,
    "mask_regular": L1Loss,
    "mse_loss": MSELoss,
}

"""LOSS_MODULE_REGISTRY = {
    "masked": mask_loss,
    "contrastive": contrastive_loss,
    "clip": contrastive_loss_clip,
    "cross_entropy": ce_loss,
    "soft_cross_entropy": sce_loss,
}"""

OUTPUT_LAYER_REGISTRY = {
    "multi_class": multiclass_classification,
    "binary_class": multiclass_classification,
    "multi_label": multilabel_classification,
}


TASK_PREPROCESSING_LAYER = {
    "supervised": SupervisedInputTransform,
    "masked": MaskInputTransforms,
    "contrastive": ContrastiveInputTransforms,
    "clip": CLIPInputTransforms,
}

INTERMEDIATE_TRANSFORM_LAYER = {"view_select": ViewSelect}


supervised_taskflow_default = [
    {
        "name": "supervised_task_preprocessing",
        "module": "supervised_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [("supervised_task_preprocessing", 0)],
    },
    {"name": "encoder", "module": "encoder", "inputs": [("pre_encoder", 0)]},
    {"name": "classifier", "module": "classifier", "inputs": [("encoder", 1)]},
]

supervised_taskflow_patchaug = [
    {
        "name": "supervised_task_preprocessing",
        "module": "supervised_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [("supervised_task_preprocessing", 0)],
    },
    {
        "name": "patch_augmentation",
        "module": "patch_augmentation",
        "inputs": [
            ("pre_encoder", 0),
            ("supervised_task_preprocessing", 1),
            ("supervised_task_preprocessing", 2),
        ],
    },
    {
        "name": "encoder",
        "module": "encoder",
        "inputs": [("patch_augmentation", 0)],
    },
    {"name": "classifier", "module": "classifier", "inputs": [("encoder", 1)]},
]

supervised_taskflow_featureaug = [
    {
        "name": "supervised_task_preprocessing",
        "module": "supervised_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [("supervised_task_preprocessing", 0)],
    },
    {"name": "encoder", "module": "encoder", "inputs": [("pre_encoder", 0)]},
    {
        "name": "feature_augmentation",
        "module": "feature_augmentation",
        "inputs": [
            ("encoder", 1),
            ("supervised_task_preprocessing", 1),
            ("supervised_task_preprocessing", 2),
        ],
    },
    {
        "name": "classifier",
        "module": "classifier",
        "inputs": [("feature_augmentation", 0)],
    },
]


masked_taskflow_default = [
    {
        "name": "masked_task_preprocessing",
        "module": "masked_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [
            ("masked_task_preprocessing", 0),
            ("masked_task_preprocessing", 1),
        ],
    },
    {
        "name": "encoder",
        "module": "encoder",
        "inputs": [("pre_encoder", 0)],  # src_pre_enccoding
    },
    {
        "name": "decoder",
        "module": "decoder",
        "inputs": [
            ("encoder", 1),  # src_encoding_hidden
            ("pre_encoder", 1),  # target_pre_encoding
            ("masked_task_preprocessing", 2),  # mask
        ],
    },
]

masked_taskflow_patchaug = [
    {
        "name": "masked_task_preprocessing",
        "module": "masked_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [
            ("masked_task_preprocessing", 0),
            ("masked_task_preprocessing", 1),
        ],
    },
    {
        "name": "patch_augmentation",
        "module": "patch_augmentation",
        "inputs": [("pre_encoder", 0), ("masked_task_preprocessing", 1)],
    },
    {
        "name": "encoder",
        "module": "encoder",
        "inputs": [("patch_augmentation", 0)],  # src_pre_enccoding
    },
    {
        "name": "decoder",
        "module": "decoder",
        "inputs": [
            ("encoder", 1),  # src_encoding_hidden
            ("pre_encoder", 1),  # target_pre_encoding
            ("masked_task_preprocessing", 2),  # mask
        ],
    },
]

masked_taskflow_featureaug = [
    {
        "name": "masked_task_preprocessing",
        "module": "masked_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [
            ("masked_task_preprocessing", 0),
            ("masked_task_preprocessing", 1),
        ],
    },
    {
        "name": "encoder",
        "module": "encoder",
        "inputs": [("pre_encoder", 0)],  # src_pre_enccoding
    },
    {
        "name": "patch_augmentation",
        "module": "patch_augmentation",
        "inputs": [("encoder", 1), ("masked_task_preprocessing", 1)],
    },
    {
        "name": "decoder",
        "module": "decoder",
        "inputs": [
            ("encoder", 1),  # src_encoding_hidden
            ("pre_encoder", 1),  # target_pre_encoding
            ("masked_task_preprocessing", 2),  # mask
        ],
    },
]


contrastive_taskflow_default = [
    {
        "name": "contrastive_task_preprocessing",
        "module": "contrastive_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [("contrastive_task_preprocessing", 0)],
    },
    {"name": "encoder", "module": "encoder", "inputs": [("pre_encoder", 0)]},
]

contrastive_taskflow_patchaug = [
    {
        "name": "contrastive_task_preprocessing",
        "module": "contrastive_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder",
        "module": "pre_encoder",
        "inputs": [("contrastive_task_preprocessing", 0)],
    },
    {
        "name": "patch_augmentation",
        "module": "patch_augmentation",
        "inputs": [("pre_encoder", 0), ("contrastive_task_preprocessing", 1)],
    },
    {
        "name": "encoder",
        "module": "encoder",
        "inputs": [("patch_augmentation", 0)],
    },
]


clip_taskflow_default = [
    {
        "name": "clip_task_preprocessing",
        "module": "clip_task_preprocessing",
        "inputs": [("_input_", "inputs")],
    },
    {
        "name": "pre_encoder_img",
        "module": "pre_encoder_img",
        "inputs": [("clip_task_preprocessing", "image")],
    },
    {
        "name": "pre_encoder_text",
        "module": "pre_encoder_text",
        "inputs": [("clip_task_preprocessing", "text")],
    },
    {
        "name": "text_encoder",
        "module": "encoder_text",
        "inputs": [("pre_encoder_text", 0)],
    },
    {
        "name": "image_encoder",
        "module": "encoder_img",
        "inputs": [("pre_encoder_image", 0)],
    },
]

TASK_FLOWS = {
    "supervised": {
        "default": supervised_taskflow_default,
        "patch_aug": supervised_taskflow_patchaug,
        "feature_aug": supervised_taskflow_featureaug,
    },
    "contrastive": {
        "default": contrastive_taskflow_default,
        "patch_aug": contrastive_taskflow_patchaug,
    },
    "masked": {
        "default": masked_taskflow_default,
        "patch_aug": masked_taskflow_patchaug,
        "feature_aug": masked_taskflow_featureaug,
    },
    "clip": {"default": clip_taskflow_default},
}


AUGMENTATION_AUGMODULE_MAP = {
    "raw": "classifier",
    "patchnet": "patchnet_augmentation",
    "feature": "feature_augmentation",
}
