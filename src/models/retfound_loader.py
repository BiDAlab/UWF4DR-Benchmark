from __future__ import annotations
import tensorflow as tf


# ---------------------------------------------------------------------
# Model construction (MATCH training backbone)
# ---------------------------------------------------------------------

def build_retfound_model(
    num_classes: int = 2,
    image_size: int = 224,
) -> tf.keras.Model:
    """
    Build RETFound exactly as used in training.

    IMPORTANT:
    Although training script references vit_large_patch16_224_mae,
    the MAE model is dynamically registered. In standalone evaluation,
    we must use vit_large_patch16_224.
    """

    try:
        import tfimm
    except ImportError as e:
        raise ImportError(
            "tfimm is required for RETFound support.\n"
            "Install it using: pip install -r requirements/retfound.txt"
        ) from e

    base_model = tfimm.create_model(
        "vit_large_patch16_224",
        nb_classes=num_classes,
        pretrained=False,
    )

    # Match training wrapper when cutmix == 0
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(factor=(0.0, 0.5)),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input")
    x = data_augmentation(inputs)
    outputs = base_model(x)

    model = tf.keras.Model(inputs, outputs, name="retfound_wrapped")

    return model


# ---------------------------------------------------------------------
# Weight loading (REQUIRED for MAE-style checkpoints)
# ---------------------------------------------------------------------

def load_retfound_weights(
    model: tf.keras.Model,
    weights_path: str
) -> tf.keras.Model:

    # Build graph first
    _ = model(tf.zeros((1, 224, 224, 3)))

    # MAE-style checkpoint loading
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def load_retfound_model(
    weights_path: str,
    num_classes: int = 2,
) -> tf.keras.Model:

    model = build_retfound_model(
        num_classes=num_classes,
        image_size=224,
    )

    model = load_retfound_weights(model, weights_path)

    return model
