"""
ViT-B/16 model definition for UWF4DR tasks.

IMPORTANT:
- ViT-B/16 models were trained using a FIXED input resolution of 448x448
  for ALL tasks (Task 1, Task 2, Task 3).
- This follows the original experimental setup used in the paper.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from vit_keras import vit


def build_vitb16(input_shape=(448, 448, 3)):
    """
    Build a ViT-B/16 model with a binary classification head.

    Parameters
    ----------
    input_shape : tuple
        Expected to be (448, 448, 3). Other sizes are not supported, as the
        model was trained exclusively at this resolution.

    Returns
    -------
    tf.keras.Model
        ViT-B/16 model with a binary classification head.
    """

    # Safety check to avoid silent misuse
    assert input_shape[0] == 448 and input_shape[1] == 448, (
        "ViT-B/16 models were trained with 448x448 input resolution only."
    )

    # Backbone
    backbone = vit.vit_b16(
        image_size=448,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    backbone.trainable = False

    # Classification head (matches training scripts)
    model = models.Sequential(
        [
            backbone,
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def load_vitb16(weights_path=None, input_shape=(448, 448, 3)):
    """
    Load a trained ViT-B/16 model (.keras) or create a dummy model if no
    weights are provided.

    Parameters
    ----------
    weights_path : str or None
        Path to a .keras model file. If None or invalid, a dummy (untrained)
        model is created.
    input_shape : tuple
        Must be (448, 448, 3).

    Returns
    -------
    tf.keras.Model
        Loaded or dummy ViT-B/16 model.
    """

    # Enforce correct input resolution
    assert input_shape[0] == 448 and input_shape[1] == 448, (
        "ViT-B/16 models expect 448x448 inputs."
    )

    if weights_path is not None and str(weights_path).endswith(".keras"):
        print(f"📦 Loading ViT-B/16 model from: {weights_path}")
        return tf.keras.models.load_model(weights_path)

    print("⚠️  No pretrained weights provided. Building dummy ViT-B/16 model.")
    return build_vitb16(input_shape=input_shape)

















import tensorflow as tf
from keras import layers, models
from vit_keras import vit


def build_vitb16(input_shape):
    """
    Build ViT-B/16 exactly as used during training.
    """

conv_base = vit.vit_b16(
    image_size=448,
    pretrained=True,
    include_top=False,
    pretrained_top=False)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(
        layers.Dense(
            128,
            activation="relu"
        )
    )
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def load_vitb16(weights_path, input_shape):
    """
    Load a trained ViT-B/16 model (.keras) or build it.
    """

    if weights_path is not None:
        return tf.keras.models.load_model(weights_path)

    return build_vitb16(input_shape)
