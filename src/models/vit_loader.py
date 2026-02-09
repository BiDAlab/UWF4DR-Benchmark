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
