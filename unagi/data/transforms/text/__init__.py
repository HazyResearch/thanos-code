from unagi.data.transforms.text.back_translate import BackTranslate
from unagi.data.transforms.text.identity import Identity
from unagi.data.transforms.text.pretrained_lm_tokenize import PretrainedLMTokenize

ALL_TRANSFORMS = {
    "PretrainedLMTokenize": PretrainedLMTokenize,
    "BackTranslate": BackTranslate,
    "Identity": Identity,
}
