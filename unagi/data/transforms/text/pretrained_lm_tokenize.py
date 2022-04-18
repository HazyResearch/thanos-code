import torch
from transformers import AutoTokenizer

from unagi.data.transforms.text.transform import UnagiTransform


class PretrainedLMTokenize(UnagiTransform):
    def __init__(
        self,
        name=None,
        prob=1.0,
        level=0,
        model="bert-base-uncased",
        padding="max_length",
        truncation=True,
        max_length=128,
    ):
        super().__init__(name, prob, level)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def transform(self, text, label, **kwargs):
        if isinstance(text, str):
            tokens = torch.LongTensor(
                self.tokenizer(
                    text,
                    padding=self.padding,
                    truncation=self.truncation,
                    max_length=self.max_length,
                )["input_ids"]
            )
        else:
            tokens = text
        return tokens, label
