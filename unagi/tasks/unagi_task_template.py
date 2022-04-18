import logging
from functools import partial

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn

logger = logging.getLogger(__name__)


def create_unagi_task(
    model_name,
    model,
    dataset_name,
    task_flow,
    loss_module,
    loss_fns,
    output_classification,
    task_metric,
    n_views,
):
    loss = loss_module
    output = output_classification

    logger.info(f"Built model: {model_name}")

    return EmmentalTask(
        name=dataset_name,
        module_pool=nn.ModuleDict({"base_model": model}),
        task_flow=task_flow,
        loss_func=partial(loss, "base_model", model, loss_fns, n_views),
        output_func=partial(output, "base_model"),
        scorer=Scorer(metrics=task_metric),
    )
