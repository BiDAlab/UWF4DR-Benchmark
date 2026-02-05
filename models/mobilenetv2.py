import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


def build_mobilenetv2(
    input_shape,
    backbone_weights="imagenet",
):
    """
    Build a MobileNetV2-based binary classification model.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input images, e.g. (448, 448, 3) or (800, 800, 3).
    backbone_weights : str or None
        Pretrained weights for the backbone.

    Returns
    -------
    model : tf.keras.Model
        Untrained MobileNetV2 model.
    """

    base = MobileNetV2(
        weights=backbone_weights,
        include_top=False,
        input_shape=input_shape,
    )

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=outputs)

    return model


def load_mobilenetv2(
    weights_path,
    input_shape,
    backbone_weights="imagenet",
):
    """
    Load a MobileNetV2 binary classification model with pretrained weights.

    Parameters
    ----------
    weights_path : str
        Path to the model file (.keras) or weights file.
    input_shape : tuple
        Shape of the input images.
    backbone_weights : str or None
        Pretrained weights for the backbone.

    Returns
    -------
    model : tf.keras.Model
        MobileNetV2 model with loaded weights.
    """

    # Full model provided
    if weights_path.endswith(".keras"):
        return tf.keras.models.load_model(weights_path)

    # Weights only
    model = build_mobilenetv2(
        input_shape=input_shape,
        backbone_weights=backbone_weights,
    )

    model.load_weights(weights_path)

    return model
