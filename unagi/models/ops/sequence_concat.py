import torch
from torch import nn


class SequenceConcat(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "sequence_concat"

    def forward(self, *args):
        return torch.cat(args, dim=1)
