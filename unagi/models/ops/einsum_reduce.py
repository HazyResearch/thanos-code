import torch
from torch import nn


class EinsumReduceDecoder(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        # NOTE: compute d_input as module instantiation time
        # d_input = sum(d_model of all encoders being fed to Classifier)
        self.attend = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: intermediate outpus from encoder. shape: (B, S, H)
        """
        x = torch.einsum("b s o, b s d -> b d", self.attend(x).softmax(-1), x)
        return x.mean(-2)
