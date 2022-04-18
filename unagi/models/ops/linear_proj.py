from torch import nn


class LinearProj(nn.Module):
    def __init__(self, d_input, d_output, **kwargs):
        super().__init__()
        self.linear_proj = nn.Linear(d_input, d_output)

    def forward(self, x):
        return self.linear_proj(x)
