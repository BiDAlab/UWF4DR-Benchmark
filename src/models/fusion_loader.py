import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------
# Dummy builder (optional)
# ---------------------------------------------------------------------

def build_fusion_mlp(input_dim: int,
                     l2_reg: float = 0.0,
                     dropout_rate: float = 0.5):
    """
    Build MLP used for feature-level fusion.

    Architecture (as in training scripts):

        Input
        → Dense(512, ReLU)
        → Dropout(0.5)
        → Dense(256, ReLU)
        → Dropout(0.5)
        → Dense(1, Sigmoid)

    Parameters
    ----------
    input_dim : int
        Dimension of concatenated feature vector.
    l2_reg : float
        L2 regularization factor (default 0.0).
        Some tasks used small L2 during training.
    dropout_rate : float
        Dropout rate (default 0.5).

    Returns
    -------
    tf.keras.Model
    """

    regularizer = (
        tf.keras.regularizers.l2(l2_reg)
        if l2_reg > 0.0 else None
    )

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizer),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid'),
    ])

    return model


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------

def load_fusion_model(weights_path: str):
    """
    Load trained fusion MLP (.keras format).

    Parameters
    ----------
    weights_path : str
        Path to .keras fusion model.

    Returns
    -------
    tf.keras.Model
    """
    model = tf.keras.models.load_model(weights_path)
    return model
