"""
Model loader for Vision Transformers (ViT).

Currently supported:
- ViT-B/16 (trained at 448x448 for all tasks)
"""

from src.models.vitb16 import load_vitb16


def load_vit_model(weights_path=None, input_shape=(448, 448, 3)):
    """
    Load a Vision Transformer model.

    Parameters
    ----------
    weights_path : str or None
        Path to a .keras model file. If None, a dummy model is created.
    input_shape : tuple
        Must be (448, 448, 3), consistent with the training setup.

    Returns
    -------
    tf.keras.Model
        Loaded or dummy ViT model.
    """

    # ViT-B/16 was trained with fixed resolution
    return load_vitb16(
        weights_path=weights_path,
        input_shape=input_shape,
    )
