import random


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

    def transform(self, pil_img, label, **kwargs):
        return pil_img, label

    def __call__(self, pil_img, label, **kwargs):
        if random.random() <= self.prob:
            return self.transform(pil_img, label, **kwargs)
        else:
            return pil_img, label

    def __repr__(self):
        return f"<Transform ({self.name}), prob={self.prob}, level={self.level}>"
