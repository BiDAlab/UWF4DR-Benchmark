from __future__ import annotations
import tensorflow as tf


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_retfound_model(num_classes: int = 2) -> tf.keras.Model:
    """
    Instantiate RETFound ViT-Large architecture (vit_large_patch16_224).

    Args:
        num_classes: number of output classes used during fine-tuning.

    Returns:
        tf.keras.Model
    """
    try:
        import tfimm
    except ImportError as e:
        raise ImportError(
            "tfimm is required for RETFound support.\n"
            "Install it using: pip install -r requirements/retfound.txt"
        ) from e

    model = tfimm.create_model(
        "vit_large_patch16_224",
        nb_classes=num_classes,
        pretrained=False,
    )

    return model


# ---------------------------------------------------------------------
# Strict weight loading (NO silent skipping)
# ---------------------------------------------------------------------

def load_retfound_weights(
    model: tf.keras.Model,
    weights_path: str
) -> tf.keras.Model:
    """
    Load fine-tuned RETFound weights (.h5) strictly.

    Args:
        model: model returned by build_retfound_model()
        weights_path: path to .h5 weights

    Returns:
        model with loaded weights
    """

    # Build graph before loading (important for tfimm models)
    _ = model(tf.zeros((1, 224, 224, 3)))

    # STRICT loading: do NOT allow silent skipping
    model.load_weights(weights_path)

    return model


def load_retfound_model(
    weights_path: str,
    num_classes: int = 2
) -> tf.keras.Model:
    """
    Convenience wrapper: build + load in one call.
    """
    model = build_retfound_model(num_classes=num_classes)
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
