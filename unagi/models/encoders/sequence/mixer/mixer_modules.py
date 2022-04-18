from torch import nn

from unagi.models.layers.blocks import FFN, PreNorm, Residual


class mixer(nn.Module):
    def __init__(self, d, n=64, dropout=0.0):
        super().__init__()
        self.f = FFN(n, n << 1)

    def forward(self, x):
        # b x p x c
        return self.f(x.transpose(1, 2)).transpose(1, 2)


# Encoder and decoder blocks


def _prenorm(d, x, drop_path=0.0):
    return Residual(d, PreNorm(d, x), drop_path=drop_path)


class mixer_encoder(nn.Module):
    def __init__(
        self,
        d,
        num_heads,
        l_max,  # should be equal to the sequence length
        mlp_dim=None,
        dropout=0.1,
        drop_path=0.0,
        head_dropout=None,
    ):
        super().__init__()

        def _pre(x):
            return _prenorm(d, x, drop_path=drop_path)

        self.mlp = _pre(mixer(d, n=l_max, dropout=dropout))
        mlp_dim = d << 1 if mlp_dim is None else mlp_dim
        self.ffn = _pre(FFN(d, mlp_dim, dropout=dropout))

    def forward(self, x, mask=None):
        x = self.mlp(x)
        return self.ffn(x)
