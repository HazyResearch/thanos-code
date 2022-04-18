import torch
from transformers import AutoTokenizer, BertModel

from unagi.models.encoders.base_sequence import SequenceModule


class BertEncoder(SequenceModule):
    def __init__(
        self,
        freeze_layers=True,
        pretrained_lm_name="bert-base-uncased",
        use_cls_token=True,
        use_all_tokens=False,
        pretrained_weights=None,
        **kwargs,
    ):
        super().__init__()
        self.f = BertModel.from_pretrained(pretrained_lm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_name)
        self.f = self.f.train()
        self.use_cls_token = use_cls_token
        self.use_all_tokens = use_all_tokens

        """if freeze_layers:
            for param in self.f.parameters():
                param.requires_grad = False"""

        self.d_model = self.f.encoder.layer[-1].output.dense.out_features
        self.padding = "max_length"
        self.truncation = True
        self.max_length = 128

    def forward(self, x):
        # tok_out = self.tokenizer(
        #     x,
        #     padding=self.padding,
        #     truncation=self.truncation,
        #     max_length=self.max_length,
        # )
        # input_ids = torch.LongTensor(tok_out["input_ids"])
        # attention_mask = torch.LongTensor(tok_out["attention_mask"])
        input_ids = x
        attention_mask = (x != 0).long()
        token_type_ids = torch.zeros_like(input_ids)

        # output = self.f(inputs_embeds=x, return_dict=True)
        output = self.f(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        if self.use_cls_token:
            # return output["pooler_output"]
            return output["last_hidden_state"][:, 0, :].squeeze(dim=1)
        else:
            return output["last_hidden_state"]
