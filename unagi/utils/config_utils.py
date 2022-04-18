import copy
import os
from collections.abc import Mapping

from unagi.configs import (
    BASE_CONFIG_PATH,
    BASE_DATASET_PATH,
    BASE_INPUT_FEATURE_PATH,
    BASE_OUTPUT_FEATURE_PATH,
    DATASET_DEFAULT_PATHS,
)
from unagi.utils.file_utils import load_yaml


def get_feature_to_type(config):
    dataset_config = config["dataset"]
    input_features = dataset_config["input_features"]
    mapping = {}
    for feat in input_features:
        mapping[feat["name"]] = feat["type"]
    return mapping


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


# TODO (ASN): modularize to have functions for each
def config_preprocessing(config: dict):
    # model_section = config["model"]
    # tasks_section = config["tasks"]
    return


def build_config(user_config: dict) -> dict:
    """
    Merges user specified config file with default config file and performs
    section pre-processing

    # Inputs
    :param root_path: (str) filepath to unagi directory
    :param user_config (dict) dictionary representationn of use config

    # Returns
    :return: prepcessed config

    """
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_cf = load_yaml(os.path.join(root_path, BASE_CONFIG_PATH))
    base_cf = merge_dict(base_cf, user_config)

    # load the dataset defaults
    default_dataset = load_yaml(os.path.join(root_path, BASE_DATASET_PATH))
    base_input_feature = load_yaml(os.path.join(root_path, BASE_INPUT_FEATURE_PATH))
    base_output_feature = load_yaml(os.path.join(root_path, BASE_OUTPUT_FEATURE_PATH))

    # merge model config
    # TODO (ASN): replace w/generic modules
    """
    base_model = load_yaml(
        os.path.join(root_path, MODEL_DEFAULT_PATHS[user_config["model"]["name"]])
    )
    base_cf["model"] = merge_dict(base_model["model"], base_cf["model"])
    """

    # merge dataset config
    base_dataset = load_yaml(
        os.path.join(root_path, DATASET_DEFAULT_PATHS[user_config["dataset"]["name"]])
    )
    base_dataset = merge_dict(default_dataset["dataset"], base_dataset["dataset"])
    base_cf["dataset"] = merge_dict(base_dataset, base_cf["dataset"])

    for key, input_feature in base_cf["dataset"]["input_features"].items():
        base_cf["dataset"]["input_features"][key] = merge_dict(
            base_input_feature, input_feature
        )
    for key, output_feature in base_cf["dataset"]["output_features"].items():
        base_cf["dataset"]["output_features"][key] = merge_dict(
            base_output_feature, output_feature
        )

    # pre_processed_config = config_preprocessing(base_cf)

    # TODO (ASN): ADD SECTION PRE-PROCESSING
    # MODEL : --> correct for all conflicting params (use_all_toksn vs use_cls_token)
    # TASKS :
    #   check that all tasks that are specified are valid
    #   check that all loss functions are valid
    #   compute any task specific computations (mask length)
    #
    return base_cf
