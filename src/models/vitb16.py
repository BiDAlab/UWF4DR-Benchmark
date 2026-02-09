import tensorflow as tf
from vit_keras import vit


def build_vitb16(input_shape):
    """
    Build ViT-B/16 binary classification model
    exactly as used during training.
    """

    if input_shape[0] != input_shape[1]:
        raise ValueError("ViT-B/16 requires square input")

    image_size = input_shape[0]

    backbone = vit.vit_b16(
        image_size=image_size,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )

    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def load_vitb16(weights_path, input_shape):
    """
    Load a trained ViT-B/16 model (.keras) or build it.
    """

    # Full model provided
    if weights_path is not None and weights_path.endswith(".keras"):
        return tf.keras.models.load_model(
            weights_path,
            compile=False,
            safe_mode=False,
        )

    # Build untrained model (dummy / testing)
    return build_vitb16(input_shape)
