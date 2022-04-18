from unagi.models.decoders.classifier import ClassificationDecoder
from unagi.models.decoders.image.resnet import ResnetDecoder
from unagi.models.decoders.image.resnet_autoencoder import (
    Resnet18Decoder,
    Resnet50Decoder,
)
from unagi.models.decoders.sequence.mixer import MixerDecoder
from unagi.models.decoders.sequence.transformer import TransformerDecoder
from unagi.models.embeddings.embeddings import (
    CategoricalEmbed,
    Conv2DEmbed,
    ConvEmbed,
    IdentityEmbed,
    LinearPatchEmbed,
    PretrainedLMEmbed,
    SquarePatchEmbed,
)
from unagi.models.encoders.image.resnet.resnet import ResnetEncoder
from unagi.models.encoders.sequence.bert.bert import BertEncoder
from unagi.models.encoders.sequence.mixer.mixer import MixerEncoder
from unagi.models.encoders.sequence.transformer.transformer import TransformerEncoder
from unagi.models.layers.patch_augmentations import (
    BrightnessLayer,
    CutoutLayer,
    InvertLayer,
    MixUpLayer,
    RotateLayer,
    SolarizeLayer,
)
from unagi.models.ops.grayscale import Grayscale
from unagi.models.ops.image_reshape import ImageReshape
from unagi.models.ops.linear_proj import LinearProj
from unagi.models.ops.pool import PoolDecoder
from unagi.models.ops.sequence_concat import SequenceConcat
from unagi.models.ops.view_concat import ViewConcat
from unagi.models.ops.view_select import ViewSelect

MODULE_DICTS = {
    "embeddings": {
        "square_patch": SquarePatchEmbed,
        "linear_patch": LinearPatchEmbed,
        "categorical": CategoricalEmbed,
        "conv2d": Conv2DEmbed,
        "conv1d": ConvEmbed,
        "pretrained_lm": PretrainedLMEmbed,
        "identity": IdentityEmbed,
        "sequence_concat": SequenceConcat,
    },
    "encoders": {
        "mixer": MixerEncoder,
        "transformer": TransformerEncoder,
        "resnet": ResnetEncoder,
        "bert": BertEncoder,
    },
    "decoders": {
        "classifier": ClassificationDecoder,
        "pool": PoolDecoder,
        "view_select": ViewSelect,
        "view_concat": ViewConcat,
        "transformer": TransformerDecoder,
        "mixer": MixerDecoder,
        "resnet": ResnetDecoder,
        "resnet18decoder": Resnet18Decoder,
        "resnet50decoder": Resnet50Decoder,
        "image_reshape": ImageReshape,
        "sequence_concat": SequenceConcat,
        "linear_proj": LinearProj,
        "grayscale": Grayscale,
    },
}

AUGMENTATION_LAYERS = {
    "patch": {
        "mixup": MixUpLayer,
        "invert": InvertLayer,
        "cutout": CutoutLayer,
        "solarize": SolarizeLayer,
        "brightness": BrightnessLayer,
        "rotate": RotateLayer,
    },
    "feature": {
        "mixup": MixUpLayer,
        "invert": InvertLayer,
        "cutout": CutoutLayer,
        "solarize": SolarizeLayer,
        "brightness": BrightnessLayer,
        "rotate": RotateLayer,
    },
}
