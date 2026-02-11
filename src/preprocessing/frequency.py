import numpy as np
import cv2
import tensorflow as tf


def center_crop(image, crop_size=(800, 800)):
    h, w, _ = image.shape
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[top:top + ch, left:left + cw]


def freq_transform_mag_clipped(image):
    """
    Compute clipped FFT magnitude (percentile 99) per channel,
    exactly as used during training.
    """
    image = image.astype(np.float32)
    freq_image = np.zeros_like(image, dtype=np.float32)

    for c in range(3):
        f = np.fft.fft2(image[..., c])
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        clip_val = np.percentile(magnitude, 99)
        magnitude = np.clip(magnitude, 0, clip_val)

        magnitude_norm = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        )
        freq_image[..., c] = magnitude_norm

    return freq_image.astype(np.float32)


def preprocess_frequency(image, target_size):
    """
    Frequency-domain preprocessing for UWF images.

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    target_size : tuple
        Final size (H, W).

    Returns
    -------
    np.ndarray
        Preprocessed frequency-domain image compatible with MobileNetV2.
    """
    image = center_crop(image, (800, 800))
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    image = freq_transform_mag_clipped(image)

    return image
