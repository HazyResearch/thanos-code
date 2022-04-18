import random

PRECISION = 3


class UnagiTransform(object):
    """Base UnagiTransform transfrom class.

    Args:
      name(str): Transformation name.
      prob(float): Transformation probability.
      level(int): Transformation level.
    """

    def __init__(self, name=None, prob=1.0, level=0):
        self.name = name if name is not None else type(self).__name__
        self.prob = prob

        assert 0 <= level <= 1.0, "Invalid level, level must be in [0, 1.0]."

        self.level = level

    def transform(self, text, label, **kwargs):
        return text, label

    def __call__(self, text, label, **kwargs):
        if random.random() <= self.get_prob():
            return self.transform(text, label, **kwargs)
        else:
            return text, label

    def __repr__(self):
        return f"<Transform ({self.name}), prob={self.prob}, level={self.level}>"

    def get_prob(self):
        if self.prob == 1:
            return self.prob
        return random.random()

    def get_level(self):
        return random.randint(0, 10 ** PRECISION) / float(10 ** PRECISION)
