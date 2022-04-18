import torch
import torch.nn.functional as F

from unagi.tasks.loss_fns.base_loss import UnagiLoss


class BatchMask(UnagiLoss):
    def __init__(self):
        super().__init__()

    def forward(self, last_layer, embs):
        # embs == output of embedding layers
        # last_layer == output of the decoder
        # Both embs and last_layer
        # batch x sentence x dims
        # for each prediction in the last layer, we assume no duplicate ftm.
        a = torch.einsum("b s d, b t d -> b s t", last_layer, embs)
        return -torch.diagonal(a.log_softmax(-1), dim1=1, dim2=2).mean()


class BatchMaskDup(UnagiLoss):
    def __init__(self, eps=1e-5):
        super().__init__()
        print("Using Batchmask")
        self.eps = eps

    def forward(self, last_layer, embs):
        # embs == output of embedding layers
        # last_layer == output of the decoder
        #
        # Both embs and last_layer
        # batch x sentence x dims
        # for each prediction in the last layer, we assume no duplicate ftm.
        def _g(x, y):
            return torch.einsum("b s d, b t d -> b s t", x, y)

        def _dupe_check(x):
            b, s, _ = x.size()
            x = F.normalize(x, dim=-1)
            mask = _g(x, x) > 1 - self.eps
            mask = mask.masked_fill(
                torch.triu(torch.ones(b, s, s, device=x.device)) > 0, False
            )
            # The mask is true, if there is a duplicate that comes before it in order.
            # As a result, only the first duplicate is counted.
            return mask.any(-1)

        a = _g(last_layer, embs)
        a = a.masked_fill(_dupe_check(embs).unsqueeze(1), -1e9)
        return -torch.diagonal(a.log_softmax(-1), dim1=1, dim2=2).mean()
