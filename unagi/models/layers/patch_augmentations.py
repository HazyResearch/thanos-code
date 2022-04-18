import random
from typing import Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torch.functional import Tensor


class MixUpAugmentation(nn.Module):
    """
    Inter-image augmentation: Computes augmentations on an individual sample
    """

    def __init__(self, p=0.3, prob_label=False, alpha=1):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.alpha = alpha
        self.prob_label = prob_label

    def forward(self, x, y):
        if self.alpha > 0.0:
            mix_ratio = np.random.beta(self.alpha, self.alpha)
        else:
            mix_ratio = 1.0

        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            perm_idxs = torch.randperm(x.shape[0])
            x_perm, y_perm = x[perm_idxs], y[perm_idxs]

            # input augmentation
            x_aug = mix_ratio * x + (1 - mix_ratio) * x_perm
            # label augmentation
            if self.prob_label:
                y_aug = mix_ratio * y + (1 - mix_ratio) * y_perm

            else:
                y_aug = torch.tensor(
                    [
                        y[idx] if np.random.random() < mix_ratio else y_perm[idx]
                        for idx in range(int(y.size(0)))
                    ]
                ).to(self.device)

        return x_aug, y_aug


class CutoutAugmentation(nn.Module):
    """
    Intra-image augmentation: masks out a random patch from image
    """

    def __init__(self, p=0.3, prob_label=False):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.prob_label = prob_label

    def forward(self, x, y):
        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            seq_dim = x.size(1)
            # randomly select patch to mask out. chooses a random patch for
            # each sample in batch
            for batch_idx in range(x.size(0)):
                patch_idx = random.sample(range(seq_dim), 1)[0]
                x_aug[batch_idx][patch_idx] = 0
        return x_aug, y_aug


class SolarizeAugmentation(nn.Module):
    """
    Pixel Level Augmentation: Inverts all tensor values above a threshold.
    Assumes input is normalized.
    """

    def __init__(self, p=0.3, threshold=0.5, prob_label=False):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.prob_label = prob_label
        self.threshold = threshold

    def forward(self, x, y):
        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            x_aug = torch.where(abs(x_aug) > self.threshold, 1 - x_aug, x_aug)
        return x_aug, y_aug


class BrightnessAugmentation(nn.Module):
    """
    Pixel Level Augmentation: Modifies brightness by increasing (or decreasing)
    the tensor value evenly across all pixels. Values are modified using the
    factor value
    """

    def __init__(self, p=0.3, factor=1.0, prob_label=False):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.prob_label = prob_label
        self.factor = factor

    def forward(self, x, y):
        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            x_aug = x_aug * self.factor
        return x_aug, y_aug


class InvertAugmentation(nn.Module):
    """
    Pixel Level Augmentation: Inverts tensor values -- assumes that
    input is normalized
    """

    def __init__(self, p=0.3, prob_label=False):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.prob_label = prob_label

    def forward(self, x, y):
        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            x_aug = 1 - x_aug
        return x_aug, y_aug


class RotateAugmentation(nn.Module):
    """
    Pixel Level Augmentation: Inverts tensor values -- assumes that input
    is normalized
    """

    def __init__(self, p=0.3, degree=90, prob_label=False):  # is label a float?
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.prob = p
        self.prob_label = prob_label
        self.degree = degree

    def forward(self, x, y):
        x_aug, y_aug = x, y
        if random.random() <= self.prob:
            bs = x.size(0)
            seq_len = x_aug.size(1)
            h_dim = x_aug.size(2)
            rot_samples = []
            for sample_idx in range(bs):
                batch_mat = x_aug[sample_idx]
                batch_mat = batch_mat.reshape(1, seq_len * h_dim)
                rot_mat = TF.rotate(batch_mat.unsqueeze(1), self.degree)
                rot_samples.append(rot_mat.reshape(1, seq_len, h_dim))
            x_aug = torch.stack(rot_samples)
        return x_aug, y_aug


class MixUpLayer(nn.Module):
    def __init__(self, prob=0.3, prob_label=False, alpha=1):  # is label a float?
        super().__init__()

        self.patch_aug = MixUpAugmentation(p=prob, prob_label=prob_label, alpha=alpha)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y


class InvertLayer(nn.Module):
    def __init__(self, prob=0.3, prob_label=False):  # is label a float?
        super().__init__()

        self.patch_aug = InvertAugmentation(p=prob, prob_label=prob_label)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y


class CutoutLayer(nn.Module):
    def __init__(self, prob=0.3, prob_label=False):  # is label a float?
        super().__init__()

        self.patch_aug = CutoutAugmentation(p=prob, prob_label=prob_label)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y


class BrightnessLayer(nn.Module):
    def __init__(self, prob=0.3, factor=1.0, prob_label=False):  # is label a float?
        super().__init__()

        self.patch_aug = BrightnessAugmentation(prob, factor, prob_label)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y


class SolarizeLayer(nn.Module):
    def __init__(self, prob=0.3, threshold=1.0, prob_label=False):  # is label a float?
        super().__init__()

        self.patch_aug = SolarizeAugmentation(prob, threshold, prob_label)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y


class RotateLayer(nn.Module):
    def __init__(self, prob=0.3, degree=90, prob_label=False):  # is label a float?
        super().__init__()

        self.patch_aug = RotateAugmentation(prob, degree=degree, prob_label=prob_label)

    def forward(self, patch_feature_dict: Dict[str, Tensor], Y: Tensor, is_train=False):
        """
        patch_aug (Dict[str, Tensor]): mapping between input feature name,
            and corresponding augmentated patch representation
        Y (Tensor): augmentated label
        """
        if is_train:
            aug_patches = {}
            for feat_name, feat_value in patch_feature_dict.items():
                feat_aug, Y_aug = self.patch_aug(feat_value, Y)
                aug_patches[feat_name] = feat_aug
            return aug_patches, Y_aug
        else:
            return patch_feature_dict, Y
