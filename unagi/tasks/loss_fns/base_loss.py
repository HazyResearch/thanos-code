import torch.nn as nn


class UnagiLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError
