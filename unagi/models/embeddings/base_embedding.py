from torch import nn


class EmbeddingModule(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
