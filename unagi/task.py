import logging
from typing import Any, Dict, Sequence

import hydra
import torch
from torch import nn

from unagi.trainer import MODULE_REGISTRY
from unagi.trainer.task import UnagiTask

logger = logging.getLogger(__name__)


def _instantiate_modules(module_config: Dict[str, Any], type: str):
    # Assert type with useful error message
    assert type in {
        "preprocessors",
        "embeddings",
        "encoders",
        "decoders",
        "losses",
    }, f"{type} is not a valid module type"

    module_dict = {}
    for uid, cfg in module_config.items():
        module_name, _target_ = cfg["module"], cfg["_target_"]
        module_kwargs = {
            k: v
            for k, v in cfg.items()
            if k
            not in ["uid", "module", "_target_", "path_to_checkpoint", "source_module"]
        }

        if isinstance(module_name, str):
            try:
                # TODO: add this import
                module = MODULE_REGISTRY[type][module_name](**module_kwargs)
            except KeyError:
                raise KeyError(f"{module} is not a valid module for {type} type")
        else:
            if isinstance(_target_, str):
                module = hydra.utils.get_method(path=_target_)(**module_kwargs)

        # load in weights from pretrained state_dict
        if "path_to_checkpoint" in cfg and cfg["path_to_checkpoint"] is not None:
            ckpt = torch.load(cfg["path_to_checkpoint"])
            source_module = uid
            if "source_module" in cfg and cfg["source_module"] is not None:
                source_module = cfg["source_module"]
            module.load_state_dict(
                {
                    ".".join(k.split(".")[3:]): v
                    for k, v in ckpt["state_dict"].items()
                    if (
                        source_module in k
                        and "model" in k
                        and k.split(".")[2] == source_module
                    )
                },
                strict=True,
            )

        # Add to all modules for this type
        module_dict[uid] = module

    return module_dict


def instantiate_modules(
    preprocessors_config: Sequence[Dict[str, Any]],
    embeddings_config: Sequence[Dict[str, Any]],
    encoders_config: Sequence[Dict[str, Any]],
    decoders_config: Sequence[Dict[str, Any]],
    losses_config: Sequence[Dict[str, Any]],
):
    """
    Instantiate all modules for the model.
    """
    module_dict, loss_dict = {}, {}
    module_dict.update(_instantiate_modules(preprocessors_config, "preprocessors"))
    module_dict.update(_instantiate_modules(embeddings_config, "embeddings"))
    module_dict.update(_instantiate_modules(encoders_config, "encoders"))
    module_dict.update(_instantiate_modules(decoders_config, "decoders"))
    loss_dict.update(_instantiate_modules(losses_config, "losses"))
    return nn.ModuleDict(module_dict), nn.ModuleDict(loss_dict)


def create_tasks(task_configs):

    return {
        name: UnagiTask(
            name=name,
            # module_dict=module_dict,
            # loss_dict=loss_dict,
            task_flow=task_config["task_flow"],
            losses=task_config["losses"],
            task_weight=task_config["task_weight"],
            metrics=task_config["metrics"],
            torchmetrics=task_config["torchmetrics"],
            callbacks=task_config["callbacks"],
            # weight=task_config["task_weight"],
        )
        for name, task_config in task_configs.items()
    }


# def create_task(config):
#     """
#     Builds Unagi tasks.
#     In Unagi, we define task as the learning criterea
#         (e.g. supervised, contrastive, masked).

#     # Inputs
#     :param config: (dict) dictionary representation of experiment config file

#     # Returns
#     :return: list of Unagi tasks
#     """
#     # dataset_desc = vars(get_dataset_config(config["dataset"]["name"]))
#     # model_params = get_model_params(config["model"], dataset_desc)
#     # model_params = get_text_embed_layer(config, model_params)
#     # task_metric = dataset_desc["TASK_METRIC"]

#     """if "contrastive" in config["tasks"]:
#         n_views = config["tasks"]["contrastive"]["contrastive_views"]
#     else:
#         n_views = 1"""

#     # shared_modules = get_module_dict_v2(config["model"]["name"], model_params)
#     # shared_modules = get_module_dict_v2(config["model"])
#     # augmentation_modules = add_augmentation_modules(config["augmentations"])
#     # shared_modules.update(augmentation_modules)

#     """loss_fns, _ = get_loss_fns(
#         config["tasks"],
#         config["model"]["label_smoothing"],
#         task_type=dataset_desc["TASK_TYPE"],
#     )"""

#     # loss_fns = get_loss_fns_v2(
#     #     config["tasks"],
#     #     config["model"]["label_smoothing"],
#     #     classification_type=,
#     #     # classification_type=dataset_desc["TASK_TYPE"],
#     # )

#     aug_type = None
#     if config["augmentations"]["patch"]["type"] is not None:
#         aug_type = "patch"
#     elif config["augmentations"]["feature"]["type"] is not None:
#         aug_type = "feature"

#     all_tasks = []

#     for task in config["tasks"]:
#         task_module_dict = get_input_transform_layers()
#         #task_module_dict.update(shared_modules)
#         module_pool = nn.ModuleDict(task_module_dict)
#         task_name = task["name"]
#         task_flow = task["task_flow"]
#         task_type = task["type"]
#         loss_module = get_loss_module(task_type, loss_fn=task["loss_fn"])
#         # output_func = get_output_layer(dataset_desc, task_type)
#         # scorer = Scorer(
#         # metrics=task_metric) if task_type == "supervised" else Scorer()
#         n_views = task["contrastive_views"] if "contrastive_views"
#           in task.keys() else 1
#         encoder_module_names = task["embed_layers"]

#         if task_type == "supervised":
#             classification_module_name = task["classification_layers"]
#             encoder_module_names = classification_module_name

#         loss_func = partial(
#             loss_module, loss_fns, aug_type, n_views, encoder_module_names,
#                task_name
#         )
#         if "weight" in task.keys():
#             weight = task["weight"]
#         else:
#             weight = 1

#         """action_outputs = None
#         if "action_outputs" in task.keys():
#             action_outputs = task["action_outputs"]"""

#         # if output_func is not None:
#         #     output_func = partial(output_func, classification_module_name[
#                           0])

#         # task = EmmentalTask(
#         #     name=task_name,
#         #     module_pool=module_pool,
#         #     task_flow=task_flow,
#         #     loss_func=loss_func,
#         #     output_func=output_func,
#         #     action_outputs=action_outputs,
#         #     scorer=scorer,
#         #     weight=weight,
#         # )
#         task = UnagiTask(
#             name=task_name,
#             module_pool=module_pool,
#             task_flow=task_flow,
#             loss_func=loss_func,
#             weight=weight,
#         )
#         all_tasks.append(task)
#     return all_tasks
