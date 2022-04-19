import numpy as np
from PIL import Image


def relu_model(X, E, W, classification=False, delta=0.1):
    """Relu model."""
    Y = W.dot(np.maximum((E.T @ X.T), 0.0)).reshape(-1)
    if classification:
        idx = abs(Y) >= delta
        X = X[idx]
        Y = Y[idx]
        return X, ((Y < 0).astype(int)).reshape(-1)
    else:
        return X, Y


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
