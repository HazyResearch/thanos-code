import torch
import torch.nn.functional as F
from torch import nn


class Transpose(nn.Module):
    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j

    def forward(self, x):
        return x.transpose(self.i, self.j)


class Truncate(nn.Module):
    def __init__(self, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length

    def forward(self, input):
        if self.max_sequence_length is not None:
            # NOTE: assumes input is of form (batch, seq_length, hidden_dim)
            #       or (batch, seq_length)
            if input.size(1) > self.max_sequence_length:
                input = input[:, 0 : self.max_sequence_length, :]
            elif input.size(1) < self.max_sequence_length:
                if len(input.size()) == 2:
                    pad = (0, self.max_sequence_length - input.size(1))
                elif len(input.size()) == 3:
                    pad = (0, 0, self.max_sequence_length - input.size(1), 0)
                input = F.pad(input, pad, mode="constant", value=0)
        return input


class PreNorm(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.f = f
        self.norm = nn.LayerNorm(d)

    def forward(self, x, **kwargs):
        return self.f(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, d, mlp_dim, out_dim=None, dropout=0.1):
        super().__init__()
        out_dim = d if out_dim is None else out_dim
        self.f = nn.Sequential(
            nn.Linear(d, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        return self.f(x, **kwargs)


# https://github.com/SHI-Labs/Compact-Transformers/blob/f6d43e50ece006b933eeb27b087a0c3cad3bc635/src/transformers.py#L90


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.keep_prob >= 1.0 or not self.training:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        return x.div(self.keep_prob) * random_tensor


class Residual(nn.Module):
    def __init__(self, d, f, trainable=False, per_channel=False, drop_path=0.0):
        super().__init__()
        _init = [1.0] * d if per_channel else [1.0]
        self.scalar = nn.Parameter(torch.tensor(_init)) if trainable else 1.0
        self.f = f
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, **kwargs):
        return self.drop_path(self.f(x, **kwargs)) + x * self.scalar


class Cat(nn.Module):
    def __init__(self, f, drop_path=0.0):
        super().__init__()
        self.f = f
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, **kwargs):
        y = self.drop_path(self.f(x, **kwargs))
        return torch.cat([x, y], dim=-1)


class Classifier(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        self.classification_layer = nn.Linear(input_dim, target_dim)

    def forward(self, x, **kwargs):
        return self.classification_layer(x)
