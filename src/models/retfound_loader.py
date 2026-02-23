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


def load_retfound_weights(
    model: tf.keras.Model,
    weights_path: str
) -> tf.keras.Model:
    """
    Load fine-tuned RETFound weights (.h5).

    Args:
        model: model returned by build_retfound_model()
        weights_path: path to .h5 weights

    Returns:
        model with loaded weights
    """
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # Force graph build
    _ = model(tf.zeros((1, 224, 224, 3)))

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






from __future__ import annotations

import tensorflow as tf


def _build_vit(num_classes: int, variant: str) -> tf.keras.Model:
    try:
        import tfimm
    except ImportError as e:
        raise ImportError(
            "tfimm is required for RETFound support.\n"
            "Install it using: pip install -r requirements/retfound.txt"
        ) from e

    model_name = "vit_large_patch16_224_mae" if variant == "mae" else "vit_large_patch16_224"
    vit = tfimm.create_model(
        model_name,
        nb_classes=num_classes,
        pretrained=False,
    )
    return vit


def _wrap_with_aug(vit: tf.keras.Model) -> tf.keras.Model:
    # Same structure used in main_finetune_fourier.py when cutmix == 0
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomContrast(factor=(0.0, 0.5)),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = aug(inputs)
    outputs = vit(x)
    return tf.keras.Model(inputs, outputs, name="retfound_wrapped")


def _force_build(model: tf.keras.Model) -> None:
    _ = model(tf.zeros((1, 224, 224, 3)))


def _h5_weight_names(h5_path: str) -> set[str]:
    # Extract saved weight names from a Keras H5 weights file
    import h5py

    names: set[str] = set()
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if hasattr(obj, "attrs") and "weight_names" in obj.attrs:
                wn = obj.attrs["weight_names"]
                for b in wn:
                    if isinstance(b, bytes):
                        names.add(b.decode("utf-8"))
                    else:
                        names.add(str(b))
        f.visititems(visit)
    return names


def _count_name_matches(model: tf.keras.Model, saved_names: set[str]) -> int:
    model_names = {w.name for w in model.weights}
    return len(model_names.intersection(saved_names))


def load_retfound_model_auto(weights_path: str, num_classes: int = 2) -> tf.keras.Model:
    saved = _h5_weight_names(weights_path)

    candidates = []
    for variant in ("plain", "mae"):
        vit = _build_vit(num_classes=num_classes, variant=variant)
        _force_build(vit)
        candidates.append((variant, False, vit))

        vit2 = _build_vit(num_classes=num_classes, variant=variant)
        wrapped = _wrap_with_aug(vit2)
        _force_build(wrapped)
        candidates.append((variant, True, wrapped))

    scored = []
    for variant, wrapped, model in candidates:
        score = _count_name_matches(model, saved)
        scored.append((score, variant, wrapped, model))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_score, best_variant, best_wrapped, best_model = scored[0]

    best_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    _force_build(best_model)

    print(f"[RETFound] auto-selected variant={best_variant}, wrapped={best_wrapped}, matched_weights={best_score}")

    return best_model
