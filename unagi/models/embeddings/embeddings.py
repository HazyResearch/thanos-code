from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoModel

from unagi.models.embeddings.base_embedding import EmbeddingModule
from unagi.models.layers.blocks import Transpose


class SquarePatchEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        patch_size: int,
        dropout=0.1,
        layernorm=True,
    ):
        super().__init__(d_input=d_input, d_model=d_model)
        self.patch_side = patch_size
        self.patch_size = patch_size ** 2 * d_input

        layers = [nn.Linear(self.patch_size, d_model)]
        if layernorm:
            layers.append(nn.LayerNorm(d_model))
        if dropout > 1e-3:
            layers.append(nn.Dropout(dropout))
        self.emb = nn.Sequential(*layers)

    def forward(self, x):
        """
        input: (B, C, S)
        output: (B, S // patch_size, d_model)
        """
        b, c, s = x.size()
        h_dim = int(sqrt(s))
        assert h_dim ** 2 == s, f"SquareEmb Only works on square images ({s} v {h_dim})"

        x = rearrange(x, "b c (h w) -> b c h w", b=b, c=c, h=h_dim, w=h_dim)
        # assert c == self.d_input, f"Patchsize expected {self.d_input} channels
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


class LinearPatchEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        patch_size,
        dropout=0.1,
        layernorm=True,
    ):
        super().__init__()
        self.d_input = d_input
        self.patch_size = patch_size * d_input

        layers = [nn.Linear(self.patch_size, d_model)]
        if layernorm:
            layers.append(nn.LayerNorm(d_model))
        if dropout > 1e-3:
            layers.append(nn.Dropout(dropout))
        self.emb = nn.Sequential(*layers)

    def forward(self, x):
        """
        input: (B, C, S)
        output: (B, S // patch_size, d_model)
        """
        b, c, s = x.size()
        # assert c == self.d_input, f"Patchsize expected {self.d_input}
        # channels got {c} channels"
        x = x.view(b, c * s // self.patch_size, self.patch_size)
        return self.emb(x)


class ConvEmbed(EmbeddingModule):
    def __init__(self, d_input, d_model, patch_size, n_layers=1, dropout=0.1):
        """
        input: (B, C, S)
        output: (B, (S - patch_size ) // (patch_size)^(n_layers+1), d_model)
        """
        super().__init__()
        layers = [
            nn.Conv1d(d_input, d_model, kernel_size=patch_size, stride=2),
            nn.Dropout(dropout),
        ] + [
            nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=2)
            for k in range(n_layers)
        ]

        self.emb = nn.Sequential(*layers)

    def forward(self, x):
        return rearrange(self.emb(x), "b c s -> b s c")


class Conv2DEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        patch_size,
        n_layers=1,
    ):
        """
        input: (B, C, S)
        output: (B, (((sqrt(S) - patch_size ) // (patch_size))+ 1)^2, d_model)
        """
        super().__init__()
        layers = [
            nn.Conv2d(
                d_input,
                d_model,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                padding=0,
                bias=False,
            ),
            nn.Flatten(2, 3),
            Transpose(-2, -1),
        ]

        self.emb = nn.Sequential(*layers)

    def forward(self, x):
        b, c, s = x.size()
        h_dim = int(sqrt(s))
        assert h_dim ** 2 == s, f"Only works on square images ({s} v {h_dim})"
        x = rearrange(x, "b c (h w) -> b c h w", b=b, c=c, h=h_dim, w=h_dim)
        return self.emb(x)

    # https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/cct.py
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CategoricalEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        embed_size,
        patch_size=1,
        n_layers=1,
        dropout=0.1,
        layernorm=False,
    ):
        """
        input: (B, 1)
        output: (B, 1, d_model)
        """
        super().__init__()
        self.embed_size = embed_size
        self.emb = nn.Linear(embed_size, d_model)

    def forward(self, x):
        # convert to one-hot
        input_one_hot = F.one_hot(x.to(torch.int64), num_classes=self.embed_size)
        return self.emb(input_one_hot.type(torch.FloatTensor))


class NumericalEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        patch_size,
        n_layers=1,
        dropout=0.1,
        layernorm=False,
    ):
        """
        input: (B, 1)
        output: (B, 1, d_model)
        """
        super().__init__()
        self.emb = nn.Linear(1, d_model)

    def forward(self, x):
        x = self.emb(x)
        x = x.unsqueeze(-2)
        return x


class IdentityEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        patch_size,
        n_layers=1,
        dropout=0.1,
        layernorm=False,
    ):
        super().__init__(d_input=d_input, d_model=d_model)

    def forward(self, x):
        return x


class PretrainedLMEmbed(EmbeddingModule):
    def __init__(
        self,
        d_input,
        d_model,
        patch_size,
        pretrained_lm_name: str = "bert-base-uncased",
    ):
        """ "
        input: (B, C, S) where C=1
        output: (B, S, d_model)
        """
        super().__init__(d_input=d_input, d_model=d_model)
        self.d_input = d_input
        self.patch_size = patch_size
        self.text_encoder = AutoModel.from_pretrained(pretrained_lm_name).embeddings
        self.embedding_dim = self.text_encoder.word_embeddings.embedding_dim
        self.projection_layer = nn.Linear(self.embedding_dim, d_model)
        self.emb = nn.Sequential(self.text_encoder, self.projection_layer)

    def forward(self, x):
        # TODO (ASN): add patching logic
        # b, c, s = x.size()
        # x = rearrange(x, "b c s -> (b c) s")  # get rid of single channel dim
        return self.emb(x)


# TODO: mean, sum, concat embeddings
