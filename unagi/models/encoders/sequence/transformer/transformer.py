import torch
from torch import nn

from unagi.models.encoders.base_sequence import SequenceModule
from unagi.models.encoders.sequence.transformer.transformer_modules import (
    MHA_Encoder,
    MHA_Encoder_Cat,
)


class TransformerEncoder(SequenceModule):
    def __init__(
        self,
        d_model,
        n_heads,
        l_max=512,
        n_layers=4,
        dropout=0.1,
        head_dropout=0.1,
        mlp_dim=None,
        tie_them_all=False,
        cat=False,
        d_cat=None,
        att_dim=None,
        learn_pos=True,
        use_cls_token=True,
        use_all_tokens=False,
        ret_cls_token=True,
        **kwargs
    ):
        super().__init__()

        if cat:
            assert d_cat is not None
            _f = []
            dim = d_model
            for _ in range(n_layers):
                layer = MHA_Encoder_Cat(
                    dim,
                    n_heads,
                    mlp_dim=mlp_dim,
                    att_dim=att_dim,
                    d_out=d_cat,
                    head_dropout=head_dropout,
                    drop_path=dropout,
                    dropout=dropout,
                )
                _f += [layer]
                dim += 2 * d_cat
            _f += [nn.LayerNorm(dim)]
            _f += [nn.Linear(dim, d_model)]
        else:

            def _block(drop_path=0.0):
                return MHA_Encoder(
                    d_model,
                    n_heads,
                    mlp_dim=mlp_dim,
                    head_dropout=head_dropout,
                    drop_path=drop_path,
                    dropout=dropout,
                )

            if tie_them_all:
                _f = [_block()] * n_layers
            else:
                _f = [
                    _block(
                        drop_path=k * dropout / (n_layers - 1) if n_layers > 1 else 0
                    )
                    for k in range(n_layers)
                ]
            _f += [nn.LayerNorm(d_model)]
        self.f = nn.Sequential(*_f)
        self.use_cls_token = use_cls_token
        self.use_all_tokens = use_all_tokens
        self.ret_cls_token = ret_cls_token
        self.learn_pos = learn_pos
        self.pe = nn.Parameter(1e-1 * torch.randn(l_max + 2, d_model).clamp(-1, 1))
        self.cls_token = (
            nn.Parameter(1e-1 * torch.randn(1, 1, d_model).clamp(-1, 1))
            if self.use_cls_token
            else None
        )

    def add_tokens(self, x):
        b, _, d = x.size()
        if self.use_cls_token:
            x = torch.cat(
                [self.cls_token.expand(b, self.cls_token.size(1), d), x],
                dim=1,
            )
        if self.learn_pos:
            x += self.pe[0 : x.size(1)]
        return x

    def forward(self, x, state=None, *args, **kwargs):
        x = self.add_tokens(x)
        x = self.f(x)
        if self.ret_cls_token:
            x = x[:, 0]
        return x
