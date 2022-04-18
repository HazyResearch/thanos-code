from copy import deepcopy

from unagi.data.transforms import ALL_TRANSFORMS
from unagi.data.transforms.image.compose import Compose


def get_transforms(
    input_features: dict,
    dataset_split: str,
    augmentations: dict,
    default_transforms: dict = {},
):
    """
    Gets list of transforms for each input feature.

    # Inputs
    :param input_features: (dict) contains all imput feature metadata including
    transform information used by this module.

    # Returns
    :return: returns a dict mapping input feature name to relevamt transforms.
    """
    ifeat_to_transforms = {}
    for name, inpt_feat in input_features.items():
        transforms_list = []
        feat_type = inpt_feat["type"]
        # key that has corresponding mapping in augmentations.raw section
        if dataset_split == "train":
            augmentation_key = (
                inpt_feat["transform"] if "transform" in inpt_feat.keys() else None
            )
            if augmentation_key is not None and augmentations is not None:
                augmentation_list = augmentations[augmentation_key]
                for aug in augmentation_list:
                    type = aug["type"]
                    aug = deepcopy(aug)
                    del aug["type"]
                    if type in ALL_TRANSFORMS[feat_type]:
                        transforms_list.append(ALL_TRANSFORMS[feat_type][type](**aug))
        # check if default transformation is specified in experiment config
        # if yes, overwrite preset dataset transformation
        default_transforms_key = (
            inpt_feat["default_transform"]
            if "default_transform" in inpt_feat.keys()
            else None
        )
        if default_transforms_key is not None and augmentations is not None:
            augmentation_list = augmentations[default_transforms_key]
            for aug in augmentation_list:
                type = "" + aug["type"]
                aug = deepcopy(aug)
                del aug["type"]
                if type in ALL_TRANSFORMS[feat_type]:
                    transforms_list.append(ALL_TRANSFORMS[feat_type][type](**aug))
        else:
            # use dataset preset transform
            if feat_type in default_transforms:
                transforms_list.extend(default_transforms[feat_type])
        composed_transforms = Compose(transforms_list)
        if inpt_feat["views"] >= 1:
            contrastive_transform = ALL_TRANSFORMS["task"]["Contrastive"]
            composed_transforms = contrastive_transform(
                composed_transforms,
                inpt_feat["views"] if dataset_split == "train" else 1,
            )
        if inpt_feat["mask"]:
            tuple_transform = ALL_TRANSFORMS["task"]["Mask"]
            mask_gen = ALL_TRANSFORMS["task"]["MaskGenerator"]
            composed_transforms = tuple_transform(
                composed_transforms,
                mask_gen(
                    1,  # task_config["contrastive"]["contrastive_views"],
                    inpt_feat["mask_length"],
                ),
            )

        ifeat_to_transforms[name] = composed_transforms

    return ifeat_to_transforms
