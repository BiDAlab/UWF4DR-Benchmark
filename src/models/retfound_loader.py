from __future__ import annotations
import tensorflow as tf


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_retfound_model(
    num_classes: int = 2,
    global_pool: bool = True,
    image_size: int = 224,
) -> tf.keras.Model:
    """
    Build RETFound exactly as in main_finetune_fourier.py.

    - global_pool=True  -> vit_large_patch16_224_mae
    - global_pool=False -> vit_large_patch16_224
    - cutmix == 0       -> wrap with data augmentation layers
    """

    try:
        import tfimm
    except ImportError as e:
        raise ImportError(
            "tfimm is required for RETFound support.\n"
            "Install it using: pip install -r requirements/retfound.txt"
        ) from e

    model_name = (
        "vit_large_patch16_224_mae"
        if global_pool
        else "vit_large_patch16_224"
    )

    # Backbone creation
    base_model = tfimm.create_model(
        model_name,
        nb_classes=num_classes,
        pretrained=False,
    )

    # Match training: cutmix == 0 -> wrap with augmentation layers
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
    """
    Load fine-tuned RETFound weights (.h5) from training script.

    Must use by_name=True and skip_mismatch=True because
    checkpoints are MAE-style and do not match Keras exactly.
    """

    # Build graph before loading weights
    _ = model(tf.zeros((1, 224, 224, 3)))

    # Required for RETFound MAE-style checkpoints
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


def load_retfound_model(
    weights_path: str,
    num_classes: int = 2,
    global_pool: bool = True,
) -> tf.keras.Model:
    """
    Convenience wrapper: build + load.
    """

    model = build_retfound_model(
        num_classes=num_classes,
        global_pool=global_pool,
        image_size=224,
    )

    model = load_retfound_weights(model, weights_path)

    return model


# ---------------------------------------------------------------------
# Feature extractor (CLS token)
# ---------------------------------------------------------------------

def build_retfound_feature_extractor(
    vit_model: tf.keras.Model
) -> tf.keras.Model:
    """
    Return a model that outputs the CLS token after the final norm layer.

    Output shape: (batch_size, embedding_dim)
    """

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dropout

    patch_embed = vit_model.get_layer("patch_embed")

    # First Dropout layer after patch embedding
    dropout_layer = None
    for layer in vit_model.layers:
        if isinstance(layer, Dropout):
            dropout_layer = layer
            break

    if dropout_layer is None:
        raise ValueError("Could not find Dropout layer in RETFound model.")

    # Collect transformer blocks dynamically
    blocks = []
    i = 0
    while True:
        try:
            blocks.append(vit_model.get_layer(f"blocks/{i}"))
            i += 1
        except ValueError:
            break

    if not blocks:
        raise ValueError("Could not find transformer blocks in RETFound model.")

    norm_layer = vit_model.get_layer("norm")

    class CLSExtractor(Model):
        def call(self, x, training=False):
            x = patch_embed(x)
            x = dropout_layer(x, training=training)

            for blk in blocks:
                x = blk(x, training=training)

            x = norm_layer(x)

            # CLS token
            return x[:, 0]

    extractor = CLSExtractor()
    extractor.build((None, 224, 224, 3))

    return extractor
