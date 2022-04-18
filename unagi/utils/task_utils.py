import copy
import importlib
import logging
from collections.abc import Mapping

from transformers import AutoModel, AutoTokenizer

# from unagi.configs import MODEL_DEFAULT_PATHS
from unagi.datasets import DATASET_CONFIG_REGISTRY
from unagi.models import AUGMENTATION_LAYERS, MODULE_DICTS
from unagi.tasks import (  # LOSS_FN_REGISTRY,
    LOSS_MODULE_REGISTRY,
    TASK_FLOWS,
    TASK_PREPROCESSING_LAYER,
)

# from unagi.utils.file_utils import load_yaml

logger = logging.getLogger(__name__)


def get_dataset_config(dataset_name):
    config = importlib.import_module(DATASET_CONFIG_REGISTRY[dataset_name])
    return config


def merge_dict(dct, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def get_module_dict_v2(model_config):
    module_dict = {}
    # root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for module_layer in ["embeddings", "encoders", "decoders"]:
        if module_layer in model_config.keys():
            for layer in model_config[module_layer]:

                name = layer["name"]
                type = layer["type"]
                """base_model = load_yaml(
                    os.path.join(root_path, MODEL_DEFAULT_PATHS[model])
                )
                layer = merge_dict(base_model["model"], layer)"""
                # md = copy.deepcopy(MODULE_DICTS[model][module_layer])
                md = copy.deepcopy(MODULE_DICTS[module_layer][type])

                del layer["name"]
                del layer["type"]
                module = md(**layer)
                module_dict[name] = module

    return module_dict


def collect_input_feature_views(task_flow):
    feat_to_view = {}
    for module in task_flow:
        for input in module["inputs"]:
            if isinstance(input[1], list):
                feat_to_view[input[1][0]] = input[1][0]
    return feat_to_view


def get_model_params(model_config, dataset_desc):
    """
    if args.use_cat:
    model_params.update({
        'cat': True,
        'd_cat': 64,
    })

    """
    # TODO (ASN): Clean this up
    assert not (
        model_config["use_cls_token"] and model_config["use_all_tokens"]
    ), "can't have both use_cls_token and use_all_tokens assigned to True"
    model_params = {
        # Set nClasses for LM model later
        "num_classes": dataset_desc["TASK_NUM_CLASS"],
        "num_features": dataset_desc["TASK_TOTAL_INPUT_FEATURES"],
    }
    model_params.update(model_config)

    if model_params["learn_pos"] and model_params["name"] in [
        "mixer",
        "hippo",
    ]:
        raise ValueError(
            "learn_pos cannot be set to True when training with hippo or mixer."
            " Please update your config file."
        )
    if model_params["use_cls_token"] and model_params["name"] in ["mixer"]:
        raise ValueError(
            "use_cls_token cannot be set to True when training with mixer."
            " Please update your config file."
        )
    if model_params["use_all_tokens"] and model_params["name"] in ["hippo"]:
        raise ValueError(
            "use_all_tokens cannot be set to True when training with hippo."
            " Please update your config file."
        )

    if model_config["use_cls_token"] and model_config["use_all_tokens"]:
        raise ValueError(
            "can't have both use_cls_token and use_all_tokens assigned to True."
            " Please update your config file."
        )

    return model_params


def get_mask_length(config, model_params=None):
    # TODO (ASN): fix to support new config
    model_config = config["model"]
    use_cls_token = (
        True
        if model_config["name"] != "mixer" and not model_config["use_all_tokens"]
        else False
    )
    use_mask = False
    if "masked" in config["tasks"].keys():
        use_mask = True
    mask_length = 0
    if use_mask:
        # TODO (ASN): check 1024 literal
        if model_config["patch_emb_type"] == "linear":
            mask_length = 1024 // model_config["patch_size"]
        if model_config["patch_emb_type"] == "square":
            mask_length = 1024 // model_config["patch_size"] ** 2
        if use_cls_token:
            mask_length += 1
    if model_params:
        model_params.update({"mask_length": mask_length})
    config["tasks"]["masked"]["mask_length"] = mask_length
    return model_params


def get_text_embed_layer(dataset_config, model_params):
    pretrained_lm_name = None
    contains_text_feature = False
    for input_feat in dataset_config["dataset"]["input_features"]:
        if input_feat["type"] == "text":
            default_text_transform = input_feat["default_transformation"]
            default_text_transforms = dataset_config["augmentations"]["raw"][
                default_text_transform
            ]
            for augmentation in default_text_transforms:
                if augmentation["type"] == "PretrainedLMTokenize":
                    pretrained_lm_name = augmentation["params"]["model"]

    if contains_text_feature and not pretrained_lm_name:
        return ValueError(
            "Dataset contains a text feature but no pretrained language model "
            "is specified in augmentations for tokenization. "
            "Please specifiy a PretrainedLMTokenize augmentation in your config."
        )

    if not pretrained_lm_name:
        return model_params
    else:
        model_params["text_encoder"] = AutoModel.from_pretrained(
            pretrained_lm_name
        ).embeddings
        tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_name)
        model_params["emb_mask_id"] = tokenizer.mask_token_id
        model_params["text_dim"] = 768 if "base" in pretrained_lm_name else 1024
    return model_params


"""def get_loss_fns_v2(
    learning_task_config,
    label_smoothing=False,
    classification_type="multi_class",
):
    # loss_fns = {"masked_loss": None, "contrastive_loss": None, "ce_loss": None}

    loss_fns = {}

    for task in learning_task_config:
        task_name, type, loss_fn = task["name"], task["type"], task["loss_fn"]
        loss_fns[task_name] = LOSS_FN_REGISTRY[loss_fn]()
        if type == "supervised":
            if label_smoothing and loss_fn == "cross_entropy":
                loss_fns[task_name] = LOSS_FN_REGISTRY["label_smoothing"](0.1)
            if classification_type == "multi_label":
                loss_fns[task_name] = LOSS_FN_REGISTRY["binary_cross_entropy_loss"]()
    return loss_fns"""


# def get_loss_fns(learning_task_config, label_smoothing=False,
#                   task_type="multi_class"):
#     loss_fns = {"masked_loss": None, "contrastive_loss": None, "ce_loss": None}

#     if "masked" in learning_task_config:
#         loss_fns["masked_loss"] = LOSS_FN_REGISTRY[
#             learning_task_config["masked"]["loss_fn"]
#         ]()
#         loss_module = LOSS_MODULE_REGISTRY["masked"]
#     if "contrastive" in learning_task_config:
#         params = learning_task_config["contrastive"]["loss_fn_params"]
#         params["type"] = learning_task_config["contrastive"]["loss_fn"]
#         loss_fns["contrastive_loss"] = LOSS_FN_REGISTRY[params["type"]](
#               **params)
#         loss_module = LOSS_MODULE_REGISTRY["contrastive"]
#     if "clip" in learning_task_config:
#         loss_fns["contrastive_loss"] = LOSS_FN_REGISTRY[
#             learning_task_config["clip"]["loss_fn"]
#         ]()
#         loss_module = LOSS_MODULE_REGISTRY["contrastive"]
#     if "supervised" in learning_task_config:
#         loss_fns["ce_loss"] = LOSS_FN_REGISTRY[
#             learning_task_config["supervised"]["loss_fn"]
#         ]()
#         if loss_fns["ce_loss"] == "soft_cross_entropy":
#             loss_module = LOSS_MODULE_REGISTRY["soft_cross_entropy"]
#         else:
#             loss_module = LOSS_MODULE_REGISTRY["cross_entropy"]

#     if (
#         label_smoothing
#         and learning_task_config["supervised"]["loss_fn"] == "cross_entropy"
#     ):
#         loss_fns["ce_loss"] = LOSS_FN_REGISTRY["label_smoothing"](0.1)

#     if task_type == "multi_label":
#         loss_fns["ce_loss"] = LOSS_FN_REGISTRY["binary_cross_entropy_loss"]()

#     return loss_fns, loss_module


# def get_output_layer(dataset_desc, task_name):
#     # Only add an output layer if task is supervised task

#     if task_name == "supervised":
#         return OUTPUT_LAYER_REGISTRY[dataset_desc["TASK_TYPE"]]
#     else:
#         return None


def get_module_dict(model_name, model_params):
    if model_name == "clip":

        md = copy.deepcopy(MODULE_DICTS[model_name])

        for md_name, module in md.items():
            enc_params = model_params[md_name]
            if "text_encoder" in model_params:
                enc_params["text_encoder"] = model_params["text_encoder"]
                enc_params["emb_mask_id"] = model_params["emb_mask_id"]
                enc_params["text_dim"] = model_params["text_dim"]
            md[md_name] = module(**enc_params)
        return md
    else:
        if model_name in MODULE_DICTS.keys():
            md = copy.deepcopy(MODULE_DICTS[model_name])
            # instantiate dict
            for md_name, module in md.items():
                md[md_name] = module(**model_params)
            return md
        else:
            # TODO (ASN) : move all config sanity checks to a seperate Config
            #  Preprocessing Step
            raise ValueError(f"No avalibale modules for this {model_name} model type")


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


def get_task_flow(task_name, model_name, aug_type=None):
    if aug_type == "patch":
        task_flow = TASK_FLOWS[task_name]["patch_aug"]
    elif aug_type == "feature":
        task_flow = TASK_FLOWS[task_name]["feature_aug"]
    else:
        task_flow = TASK_FLOWS[task_name]["default"]

    # TODO (ASN): FIX UGLY LOGIC

    if task_name == "masked" and model_name in ["mixer", "hippo"]:
        modified_task_flow = []
        for task_module in task_flow:
            if task_module["name"] == "encoder":
                modified_task_module = copy.deepcopy(task_module)
                modified_task_module["inputs"] = [
                    ("pre_encoder", 0),
                    ("masked_task_preprocessing", 2),
                ]
                modified_task_flow.append(modified_task_module)
            elif task_module["name"] == "decoder":
                modified_task_module = copy.deepcopy(task_module)
                modified_task_module["inputs"] = [
                    ("encoder", 0),
                    ("encoder", 1),
                ]
                modified_task_flow.append(modified_task_module)

            else:
                modified_task_flow.append(task_module)
        return modified_task_flow
    else:
        return task_flow


def get_loss_module(task_name, loss_fn):
    if task_name == "masked":
        return LOSS_MODULE_REGISTRY["masked"]
    elif task_name == "contrastive":
        return LOSS_MODULE_REGISTRY["contrastive"]
    elif task_name == "clip":
        return LOSS_MODULE_REGISTRY["clip"]
    elif task_name == "supervised":
        if loss_fn == "soft_cross_entropy":
            return LOSS_MODULE_REGISTRY["soft_cross_entropy"]
        else:
            return LOSS_MODULE_REGISTRY["cross_entropy"]
    else:
        raise ValueError(f"{task_name} has no associated loss module")


def get_input_transform_layers():
    task_preprocessing_layers = {}
    for task_name, task_module in TASK_PREPROCESSING_LAYER.items():
        task_preprocessing_layers[f"{task_name}_task_preprocessing"] = task_module()
    return task_preprocessing_layers


def add_augmentation_modules(augmentation_config):
    # patch_augmentation, feature_augmentation
    keys = ["patch", "feature"]

    augmentation_modules = {}

    for layer_type in keys:
        aug_type = augmentation_config[layer_type]["type"]
        if aug_type is not None:
            params = augmentation_config[layer_type]["params"]
            layer_module = AUGMENTATION_LAYERS[layer_type][aug_type]
            module = layer_module(**params)
            module_name = (
                "patch_augmentation"
                if layer_type == "patch"
                else "feature_augmentation"
            )
            augmentation_modules[module_name] = module
    return augmentation_modules
