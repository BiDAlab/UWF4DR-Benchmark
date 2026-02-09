from __future__ import annotations

from typing import Optional, Dict, Any
import os

import tensorflow as tf
from tensorflow.keras import layers, models

from vit_keras import vit


# ----------------------------
# Preprocess
# ----------------------------
def preprocess_input_vit(x: tf.Tensor) -> tf.Tensor:
    """
    Match your training preprocessing:
        x = (x / 127.5) - 1.0

    Expects x in [0, 255] (uint8/float32). Returns float32 in [-1, 1].
    """
    x = tf.cast(x, tf.float32)
    return (x / 127.5) - 1.0


# ----------------------------
# Custom objects for vit_keras==0.1.2
# ----------------------------
def _vit_custom_objects_v012() -> Dict[str, Any]:
    """
    vit_keras==0.1.2 often requires explicit custom_objects when loading .keras.
    We register the most common layer classes used by vit_keras models.
    """
    custom: Dict[str, Any] = {}

    # vit_keras 0.1.2 typically defines these in vit_keras.layers
    try:
        from vit_keras import layers as vit_layers  # type: ignore

        for name in ["ClassToken", "AddPositionEmbs", "MultiHeadSelfAttention", "TransformerBlock"]:
            if hasattr(vit_layers, name):
                custom[name] = getattr(vit_layers, name)
    except Exception:
        pass

    # Fallback: sometimes symbols can be reachable via vit_keras.vit depending on install
    try:
        from vit_keras import vit as vit_mod  # type: ignore

        for name in ["ClassToken", "AddPositionEmbs", "MultiHeadSelfAttention", "TransformerBlock"]:
            if hasattr(vit_mod, name) and name not in custom:
                custom[name] = getattr(vit_mod, name)
    except Exception:
        pass

    return custom


# ----------------------------
# Model builder
# ----------------------------
def build_vitb16(
    input_shape=(448, 448, 3),
    image_size: int = 448,
    use_l2: bool = True,
    l2_value: float = 1e-4,
    backbone_trainable: bool = False,
) -> tf.keras.Model:
    """
    Builds a ViT-B/16 binary classifier like in your training scripts:

        conv_base = vit.vit_b16(
            image_size=448, pretrained=True, include_top=False, pretrained_top=False
        )
        conv_base.trainable = False (first stage)
        model = Sequential([Input, conv_base, Dense(128,relu,[optional L2]), Dense(1,sigmoid)])

    Notes:
    - Task1/Task2 used L2(1e-4) on Dense(128)
    - Task3 did NOT use L2 in code
    """
    conv_base = vit.vit_b16(
        image_size=image_size,
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    conv_base.trainable = backbone_trainable

    reg = tf.keras.regularizers.l2(l2_value) if use_l2 else None

    model = models.Sequential(name="vitb16_classifier")
    model.add(tf.keras.Input(shape=input_shape))
    model.add(conv_base)
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=reg))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# ----------------------------
# Loader
# ----------------------------
def load_vit_model(
    weights_path: Optional[str],
    input_shape=(448, 448, 3),
    image_size: int = 448,
    task: Optional[str] = None,
    use_l2: Optional[bool] = None,
) -> tf.keras.Model:
    """
    Load a trained ViT model for inference/evaluation.

    Behavior:
    - If weights_path is None -> returns a fresh (dummy) model architecture.
    - If weights_path endswith .keras/.h5/.hdf5 -> loads full model via load_model
      (retrying with custom_objects for vit_keras==0.1.2).
    - Otherwise -> builds architecture and loads weights via model.load_weights().

    Params:
    - task: "task1"/"task2"/"task3" to match head regularization
            (task3 -> no L2, task1/2 -> L2).
    - use_l2: overrides task inference if provided.
    """
    if use_l2 is None:
        use_l2 = False if (task is not None and task.lower() == "task3") else True

    if weights_path is None:
        return build_vitb16(
            input_shape=input_shape,
            image_size=image_size,
            use_l2=use_l2,
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"ViT model/weights not found: {weights_path}")

    lower = weights_path.lower()
    is_full_model = lower.endswith(".keras") or lower.endswith(".h5") or lower.endswith(".hdf5")

    if is_full_model:
        # Try plain load first; if it fails, retry with vit_keras 0.1.2 custom objects
        try:
            return tf.keras.models.load_model(weights_path)
        except Exception as e1:
            try:
                return tf.keras.models.load_model(
                    weights_path, custom_objects=_vit_custom_objects_v012()
                )
            except Exception as e2:
                raise RuntimeError(
                    "Failed to load ViT saved model. "
                    "Tried without and with vit_keras==0.1.2 custom_objects.\n"
                    f"First error: {e1}\nSecond error: {e2}"
                ) from e2

    # weights-only path
    model = build_vitb16(
        input_shape=input_shape,
        image_size=image_size,
        use_l2=use_l2,
    )
    model.load_weights(weights_path)
    return model


# Alias to keep import names stable in eval scripts
load_vitb16 = load_vit_model
