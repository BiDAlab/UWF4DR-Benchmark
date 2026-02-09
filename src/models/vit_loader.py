from src.models.vitb16 import load_vitb16


def load_vit_model(weights_path, input_shape):
    """
    Load a ViT-B/16 model.
    """

    return load_vitb16(
        weights_path=weights_path,
        input_shape=input_shape,
    )
