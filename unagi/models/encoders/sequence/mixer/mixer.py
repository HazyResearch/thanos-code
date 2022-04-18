from torch import nn

from unagi.models.encoders.base_sequence import SequenceModule
from unagi.models.encoders.sequence.mixer.mixer_modules import mixer_encoder


class MixerEncoder(SequenceModule):
    def __init__(
        self,
        d_model,
        n_heads,
        l_max,  # can be computed based on embedding
        n_layers=4,
        dropout=0.1,
        head_dropout=0.1,
        mlp_dim=None,
        tie_them_all=False,
        **kwargs,
    ):
        super().__init__()

        def _block():
            return mixer_encoder(
                d_model,
                n_heads,
                l_max=l_max,
                mlp_dim=mlp_dim,
                head_dropout=head_dropout,
                dropout=dropout,
            )

        _f = (
            [_block()] * n_layers
            if tie_them_all
            else [_block() for k in range(n_layers)]
        )

        _f += [nn.LayerNorm(d_model)]
        self.f = nn.Sequential(*_f)

        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.head_dropout = head_dropout
        self.dropout = dropout
        self.n_layers = n_layers
        self.tie_them_all = tie_them_all

    def forward(self, x, state=None, mask=None, *args, **kwargs):
        # print(f"px={px.size()} mask={mask.size()}")
        if mask is not None:
            mask = self.truncate(mask)
        x = x.masked_fill(~mask.unsqueeze(-1), 0) if mask is not None else x
        x = self.f(x)
        return x, state
