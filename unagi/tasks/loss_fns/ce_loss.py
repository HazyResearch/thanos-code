from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from unagi.tasks.loss_fns.base_loss import UnagiLoss


class SoftCrossEntropyLoss(UnagiLoss):
    """Calculate the CrossEntropyLoss with soft targets.
    :param weight: Weight to assign to each of the classes. Default: None
    :type weight: list of float
    :param reduction: The way to reduce the losses: 'none' | 'mean' | 'sum'.
        'none': no reduction,
        'mean': the mean of the losses,
        'sum': the sum of the losses.
    :type reduction: str
    """

    def __init__(self, weight: List[float] = None, reduction: str = "mean"):
        super().__init__()
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", torch.Tensor(weight))

        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # type:ignore
        """Calculate the loss.
        :param input: prediction logits
        :param target: target probabilities
        :return: loss
        """
        n, k = input.shape
        losses = input.new_zeros(n)

        for i in range(k):
            cls_idx = input.new_full((n,), i, dtype=torch.long)
            loss = F.cross_entropy(input, cls_idx, reduction="none")
            if self.weight is not None:
                loss = loss * self.weight[i]
            losses += target[:, i].float() * loss

        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()
        elif self.reduction != "none":
            raise ValueError(f"Unrecognized reduction: {self.reduction}")

        return losses


class LabelSmoothing(UnagiLoss):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        #         nll_loss = -logprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        # smooth_loss = smooth_loss.unsqueeze(-1)  # added
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
