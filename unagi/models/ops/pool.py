from torch import nn


class PoolDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # NOTE: compute d_input as module instantiation time
        # d_input = sum(d_model of all encoders being fed to Classifier)

    def forward(self, x):
        """
        x: intermediate outpus from encoder. shape: (B, S, H)
        """
        return x.mean(-2)
