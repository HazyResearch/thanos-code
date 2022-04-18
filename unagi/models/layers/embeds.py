from math import sqrt

from einops import rearrange
from torch import nn


class Transpose(nn.Module):
    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j

    def forward(self, x):
        return x.transpose(self.i, self.j)


class SquareEmb(nn.Module):
    def __init__(self, in_d, d, _patch_size, dropout=0.1, layernorm=True):
        super(SquareEmb, self).__init__()
        self.in_d = in_d
        self.patch_side = _patch_size
        self.patch_size = _patch_size ** 2 * in_d

        layers = []
        layers.append(nn.Linear(self.patch_size, d))
        if layernorm:
            layers.append(nn.LayerNorm(d))
        if dropout > 1e-3:
            layers.append(nn.Dropout(dropout))

        self.emb = nn.Sequential(*layers)

    def forward(self, x):
        b, c, n = x.size()
        rt_n = int(sqrt(n))
        # assert rt_n**2 == n, f"Only works on square images ({n} v {rt_n})"
        x = rearrange(x, "b c (h w) -> b c h w", b=b, c=c, h=rt_n, w=rt_n)
        # assert c == self.in_d, f"Patchsize expected {self.in_d} channels
        # got {c} channels"
        patches = rearrange(
            x,
            "b c (h p) (w q) -> b (h w) (p q c)",
            p=self.patch_side,
            q=self.patch_side,
        )
        return self.emb(patches)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class LinearEmb(nn.Module):
    def __init__(self, in_d, d, _patch_size, dropout=0.1):
        super().__init__()
        self.in_d = in_d
        self.patch_size = _patch_size * in_d
        self.emb = nn.Sequential(nn.Linear(self.patch_size, d), nn.LayerNorm(d))

    def forward(self, x):
        b, c, n = x.size()
        # assert c == self.in_d, f"Patchsize expected {self.in_d}
        # channels got {c} channels"
        xx = x.view(b, c * n // self.patch_size, self.patch_size)

        return self.emb(xx)


class ConvEmb(nn.Module):
    def __init__(self, in_d, d, patch_size, nLayers=1, dropout=0.1):
        super(ConvEmb, self).__init__()
        _f = [
            nn.Conv1d(in_d, d, kernel_size=patch_size, stride=patch_size),
            nn.Dropout(dropout),
        ] + [
            nn.Conv1d(d, d, kernel_size=patch_size, stride=patch_size)
            for k in range(nLayers)
        ]
        self.f = nn.Sequential(*_f)

    def forward(self, x):
        return rearrange(self.f(x), "b c s -> b s c")


class Conv2DEmb(nn.Module):
    def __init__(self, in_d, d, patch_size):
        super().__init__()
        emb = nn.Conv2d(
            in_d,
            d,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=0,
            bias=False,
        )
        # Add batch batch or layernorm?
        self.f = nn.Sequential(emb, nn.Flatten(2, 3), Transpose(-2, -1))

    def forward(self, x):
        b, c, n = x.size()
        rt_n = int(sqrt(n))
        assert rt_n ** 2 == n, f"Only works on square images ({n} v {rt_n})"
        x = rearrange(x, "b c (h w) -> b c h w", b=b, c=c, h=rt_n, w=rt_n)
        return self.f(x)

    # https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/cct.py
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


emb_type = {
    "square": SquareEmb,
    "linear": LinearEmb,
    "conv": ConvEmb,
    "conv2d": Conv2DEmb,
}
