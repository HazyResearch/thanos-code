import torch
from einops import rearrange
from torch import nn

from unagi.models.layers.blocks import FFN, Cat, PreNorm, Residual


class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, head_dropout=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** (-0.5)

        # All the dropouts
        self.out_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.head_dropout = (
            nn.Dropout2d(head_dropout) if head_dropout is not None else None
        )

    def forward(self, q, k, v, mask=None):
        attn = torch.einsum("b s h d, b t h d -> b s h t", q, k).mul(self.scale)
        if self.head_dropout is not None:
            attn = self.head_dropout(attn)
        if mask is not None:
            attn[~mask] = -1e9
        attn = attn.softmax(dim=-1)
        # attn is batch x sentence x head x sentence
        # v is    batch x sentence x head x dim
        ret = torch.einsum("b s h d, b t h s -> b t h d", v, self.attn_dropout(attn))
        return self.out_dropout(rearrange(ret, "b t h d -> b t (h d)"))

    # Initialization from
    # https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/cct.py
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# An optimization to fuse keys-and-values


class MHA_fused(nn.Module):
    def __init__(
        self, d_model, n_heads, dropout=0.1, head_dropout=None, d_att=None, d_out=None
    ):
        """d_att: dimension of Q, K, V matrices (must be multiple of n_heads)
        d_out: output dimension of module"""
        super().__init__()

        if d_att is None:
            d_att = d_model
        if d_out is not None:
            self.proj = nn.Linear(d_att, d_out)
        else:
            self.proj = nn.Identity()
        self.qkv = nn.Linear(d_model, 3 * d_att, bias=False)
        self.f = MHA(d_att, n_heads, dropout=dropout, head_dropout=head_dropout)
        self.n_heads = n_heads

    def forward(self, x, mask=None):
        # Now batch x sentence x 3*dim
        qkv = rearrange(self.qkv(x), "b s (k h d) -> b s k h d", k=3, h=self.n_heads)
        return self.proj(self.f(*torch.unbind(qkv, dim=2), mask=mask))


class MHA_split(nn.Module):
    def __init__(self, d_model, n_heads, out_d=None, dropout=0.1, head_dropout=None):
        super().__init__()
        out_d = d_model if out_d is None else out_d
        assert (d_model % n_heads == 0) and (out_d % n_heads == 0), (
            f"The input {d_model} and output {out_d} dimensions must be multiplies of"
            f" {n_heads}"
        )

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, out_d, bias=False)
        self.f = MHA(d_model, n_heads, dropout=dropout, head_dropout=head_dropout)
        self.n_heads = n_heads

    def forward(self, q, k, v, mask=None):
        def _r(f, x):
            return rearrange(f(x), "b s (h d) -> b s h d", h=self.n_heads)

        return self.f(_r(self.q, q), _r(self.k, k), _r(self.v, v), mask=mask)


#
# Encoder and decoder blocks
#


def _prenorm(d_model, x, drop_path=0.0):
    return Residual(d_model, PreNorm(d_model, x), drop_path=drop_path)


class MHA_Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_dim=None,
        dropout=0.1,
        drop_path=0.0,
        head_dropout=None,
    ):
        super().__init__()

        def _pre(x):
            return _prenorm(d_model, x, drop_path=drop_path)

        # out_dim = d_model if out_dim is None else out_dim
        self.mha = _pre(
            MHA_fused(d_model, n_heads, dropout=dropout, head_dropout=head_dropout)
        )
        mlp_dim = d_model << 1 if mlp_dim is None else mlp_dim
        self.ffn = _pre(FFN(d_model, mlp_dim, dropout=dropout))

    def forward(self, x, mask=None):
        x = self.mha(x, mask=mask)
        return self.ffn(x)


def _cat(d_model, x, drop_path=0.0):
    return Cat(PreNorm(d_model, x), drop_path=drop_path)


class MHA_Encoder_Cat(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_dim=None,
        dropout=0.1,
        drop_path=0.0,
        head_dropout=None,
        d_out=None,
        att_dim=None,
    ):
        super().__init__()

        def _pre(x):
            return _cat(d_model, x, drop_path=drop_path)

        d_out = d_model if d_out is None else d_out
        d_att = 4 * d_out if att_dim is None else att_dim
        mha = MHA_fused(
            d_model,
            n_heads,
            dropout=dropout,
            head_dropout=head_dropout,
            d_att=d_att,
            d_out=d_out,
        )
        self.mha = _cat(d_model, mha, drop_path)
        # self.mha = _pre(
        #   MHA_fused(d_model, n_heads, dropout=dropout,
        #   head_dropout=head_dropout, d_att=4*d_out, d_out=d_out))
        #
        # self.mha = Cat(
        #   PreNorm(d_model, MHA_fused(
        #       d_model, n_heads, dropout=dropout, head_dropout=head_dropout,
        #       d_att=4*d_out, d_out=d_out)), drop_path=drop_path)
        mlp_dim = 2 * d_out if mlp_dim is None else mlp_dim
        mlp = FFN(d_model + d_out, mlp_dim, out_dim=d_out, dropout=dropout)
        self.ffn = _cat(d_model + d_out, mlp, drop_path)

    def forward(self, x, mask=None):
        x = self.mha(x, mask=mask)
        return self.ffn(x)


class MHA_Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_dim=None,
        dropout=0.1,
        drop_path=0.0,
        head_dropout=None,
    ):
        super().__init__()

        def _pre(x):
            return _prenorm(d_model, x, drop_path=drop_path)

        self._mha1 = _pre(
            MHA_fused(d_model, n_heads, dropout=dropout, head_dropout=head_dropout)
        )
        self._mha2 = _pre(
            MHA_split(d_model, n_heads, dropout=dropout, head_dropout=head_dropout)
        )

        mlp_dim = d_model << 1 if mlp_dim is None else mlp_dim
        self.ffn = _pre(FFN(d_model, mlp_dim, dropout=dropout))

    def forward(self, x, e_outputs, src_mask=None, tgt_mask=None):
        x = self._mha1(x, mask=tgt_mask)
        x = self._mha2(x, k=e_outputs, v=e_outputs, mask=src_mask)
        return self.ffn(x)
