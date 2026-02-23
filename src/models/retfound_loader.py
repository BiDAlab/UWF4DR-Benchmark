from __future__ import annotations

import os
import tempfile
from typing import Optional

import tensorflow as tf


# ---------------------------------------------------------------------
# Model construction (UNCHANGED behavior for the spatial path)
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


def load_retfound_weights(model: tf.keras.Model, weights_path: str) -> tf.keras.Model:
    """
    Load fine-tuned RETFound weights (.h5).

    IMPORTANT: This keeps the original loading behavior used in the spatial
    evaluation path to avoid breaking what already worked.
    """
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    _ = model(tf.zeros((1, 224, 224, 3)))  # force build
    return model


def load_retfound_model(weights_path: str, num_classes: int = 2) -> tf.keras.Model:
    """
    Convenience wrapper: build + load in one call.
    """
    model = build_retfound_model(num_classes=num_classes)
    model = load_retfound_weights(model, weights_path)
    return model


# ---------------------------------------------------------------------
# Feature extractor (CLS token) (UNCHANGED)
# ---------------------------------------------------------------------

def build_retfound_feature_extractor(vit_model: tf.keras.Model) -> tf.keras.Model:
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
        def call(self, x, training: bool = False):
            x = patch_embed(x)
            x = dropout_layer(x, training=training)
            for blk in blocks:
                x = blk(x, training=training)
            x = norm_layer(x)
            return x[:, 0]  # CLS token

    extractor = CLSExtractor()
    extractor.build((None, 224, 224, 3))
    return extractor


# ---------------------------------------------------------------------
# Frequency-only helper: handle nested H5 structure
#   Fourier checkpoints have top-level groups like:
#     ['input_1', 'top_level_model_weights', 'vit_large_patch16_224']
#   Spatial checkpoints are usually "flat" at root.
#
# IMPORTANT: This function is intended to be used ONLY by eval_retfound.py
# when args.domain == "frequency", so it cannot affect models that already
# worked correctly.
# ---------------------------------------------------------------------

def _copy_group_recursive(src_group, dst_group) -> None:
    """
    Deep-copy an HDF5 group (datasets + subgroups + attrs).
    """
    # Copy group attributes
    for ak, av in src_group.attrs.items():
        dst_group.attrs[ak] = av

    for key, item in src_group.items():
        import h5py  # local import

        if isinstance(item, h5py.Group):
            new_group = dst_group.create_group(key)
            _copy_group_recursive(item, new_group)
        else:
            # dataset
            src_group.copy(key, dst_group)


def load_retfound_model_frequency(weights_path: str, num_classes: int = 2) -> tf.keras.Model:
    """
    Build RETFound exactly like in main_finetune_fourier.py:
        Input -> vit_large_patch16_224

    Then load weights by name.
    """
    # Build base ViT
    vit = build_retfound_model(num_classes=num_classes)

    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = vit(inputs)
    model = tf.keras.Model(inputs, outputs)

    _ = model(tf.zeros((1, 224, 224, 3)))

    model.load_weights(weights_path, by_name=True)

    return model
