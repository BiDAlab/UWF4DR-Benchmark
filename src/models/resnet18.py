import tensorflow as tf
from keras import layers, models
from classification_models.tfkeras import Classifiers


def build_resnet18(input_shape):
    """
    Build ResNet18 exactly as used during training.
    """

    ResNet18, _ = Classifiers.get('resnet18')

    conv_base = ResNet18(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(
        layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )
    )
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def load_resnet18(weights_path, input_shape):
    """
    Load a trained ResNet18 model (.keras) or build it.
    """

    if weights_path is not None:
        return tf.keras.models.load_model(weights_path)

    return build_resnet18(input_shape)
