import torch
from torch import nn


class ViewConcat(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "view_concat"

    def forward(self, *args):
        return torch.stack(args, dim=1)
