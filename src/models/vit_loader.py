import tensorflow as tf


def load_vit_model(weights_path, input_shape):
    """
    Load a ViT-B/16 model trained for binary classification.

    Parameters
    ----------
    weights_path : str
        Path to the full .keras model file.
    input_shape : tuple
        Expected input shape, e.g. (448, 448, 3).
        Included for interface consistency with CNN loaders.

    Returns
    -------
    tf.keras.Model
    """
    if not weights_path.endswith(".keras"):
        raise ValueError(
            "ViT models must be provided as full .keras files"
        )

    model = tf.keras.models.load_model(
        weights_path,
        compile=False,
        safe_mode=False,
    )

    return model
