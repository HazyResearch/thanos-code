from torch import nn

from unagi.models.encoders.base_sequence import SequenceModule
from unagi.models.encoders.sequence.transformer.transformer_modules import MHA_Decoder


class TransformerDecoder(SequenceModule):
    def __init__(self, d_model, n_heads, dropout=0.1, head_dropout=0.1, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MHA_Decoder(d_model, n_heads, dropout=dropout, head_dropout=head_dropout)]
        )

    def forward(self, x, target, state=None, mask=None, *args, **kwargs):
        for b in self.blocks:
            tgt = b(target, x, src_mask=mask, tgt_mask=mask)
        return tgt
