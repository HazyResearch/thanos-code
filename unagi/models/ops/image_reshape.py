from einops import rearrange
from torch import nn


class ImageReshape(nn.Module):
    def __init__(self, d_input, output_height, output_width, **kwargs):
        super().__init__()
        self.name = "view_select"
        self.d_input = d_input
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, input):
        embs = rearrange(
            input, "... (h w) -> ... h w", h=self.output_height, w=self.output_width
        )
        return embs
