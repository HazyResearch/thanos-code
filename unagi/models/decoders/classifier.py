import torch
from torch import nn


class ClassificationDecoder(nn.Module):
    def __init__(self, d_input, d_output, **kwargs):
        super().__init__()
        # NOTE: compute d_input as module instantiation time
        # d_input = sum(d_model of all encoders being fed to Classifier)
        self.classification_layer = nn.Linear(d_input, d_output)

    def forward(
        self,
        *final_outs,
    ):
        """
        final_outs List[Tensor]: intermediate outputs from encoder. shape: (B, S, H)
        """
        fx = torch.cat(final_outs, dim=-1)
        return self.classification_layer(fx)
