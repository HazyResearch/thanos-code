from einops import rearrange
from torch import nn


class ViewSelect(nn.Module):
    def __init__(self, view_idx, n_views, **kwargs):
        super().__init__()
        self.name = "view_select"
        self.view_idx = view_idx
        self.n_views = n_views

    def forward(self, input):
        embs = rearrange(input, "(b v) ... -> b v ...", v=self.n_views)
        embs = embs[:, self.view_idx, ...]
        return embs
