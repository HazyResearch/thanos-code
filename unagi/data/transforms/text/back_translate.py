import math
import os
import pickle
from zipfile import ZipFile

import numpy as np

from unagi.data.transforms.text.transform import UnagiTransform


class BackTranslate(UnagiTransform):
    def __init__(self, name=None, prob=1.0, level=0, select_prob=0.5):
        super().__init__(name, prob, level)
        self.select_prob = select_prob
        # Check data exists or not.
        if not os.path.exists("data/seq2seq.pkl"):
            with ZipFile("data/seq2seq.pkl.zip", "r") as zip:
                zip.extractall("data/")
        data = open("data/seq2seq.pkl", "rb")
        self.seq2seq = pickle.load(data)

    def transform(self, text, label, **kwargs):
        ori_text = " ".join(text).strip()
        num_sents = len(self.seq2seq[ori_text][0])
        select_prob = np.random.random(size=(num_sents,))

        new_sents = [
            self.seq2seq[ori_text][0][i]
            if select_prob[i] > self.select_prob
            else self.seq2seq[ori_text][1][i]
            for i in range(num_sents)
        ]
        new_text = " ".join(new_sents).strip()

        return (self.replace_with_length_check(ori_text, new_text).split(" "), label)

    def replace_with_length_check(
        self, ori_text, new_text, use_min_length=10, use_max_length_diff_ratio=0.5
    ):
        """Use new_text if the text length satisfies several constraints."""
        if len(ori_text) < use_min_length or len(new_text) < use_min_length:
            return ori_text
        length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
        if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
            return ori_text
        return new_text
