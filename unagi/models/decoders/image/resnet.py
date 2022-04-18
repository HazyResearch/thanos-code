import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50  # noqa: F401


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        decoder_hidden_dim,
        decoder_projection_dim,
        model="resnet18",
        d_model=None,
        **kwargs,
    ):
        super().__init__()
        # self.d_model = model

        if not self.d_model:
            encoder = eval(model)()
            self.d_model = encoder.fc.in_features

        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_projection_dim),
        )

    def forward(self, x):
        return self.decoder(x)
