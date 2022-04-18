from unagi.models import MODULE_DICTS
from unagi.tasks import LOSS_MODULE_REGISTRY, TASK_PREPROCESSING_LAYER

MODULE_REGISTRY = {
    "preprocessors": TASK_PREPROCESSING_LAYER,
    "losses": LOSS_MODULE_REGISTRY,
    **MODULE_DICTS,
}
