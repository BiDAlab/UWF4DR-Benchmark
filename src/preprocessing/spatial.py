import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter


def center_crop(image, crop_size):
    h, w, _ = image.shape
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[top:top + ch, left:left + cw]


def color_normalization(image, blur_radius=9, amplification_factor=4, offset=128):
    r, g, b = image.split()

    def normalize(channel):
        blurred = channel.filter(ImageFilter.GaussianBlur(blur_radius))
        c = np.asarray(channel, np.float32)
        b = np.asarray(blurred, np.float32)
        c = np.clip((c - b) * amplification_factor + offset, 0, 255)
        return Image.fromarray(c.astype(np.uint8), mode="L")

    r = normalize(r)
    g = normalize(g)
    b = normalize(b)

    return Image.merge("RGB", (r, g, b))


def preprocess_spatial(image, target_size):
    """
    Spatial preprocessing for UWF fundus images.

    Parameters
    ----------
    image : np.ndarray
        RGB image as numpy array.
    target_size : tuple
        Final spatial size (H, W).

    Returns
    -------
    np.ndarray
        Preprocessed image ready for MobileNetV2.
    """
    image = center_crop(image, crop_size=(800, 800))
    image = tf.image.resize(image, target_size).numpy().astype(np.uint8)

    image = Image.fromarray(image)
    image = color_normalization(image)

    image = np.asarray(image)

    return image.astype(np.float32)
