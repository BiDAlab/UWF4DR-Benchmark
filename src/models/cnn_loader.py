from src.models.mobilenetv2 import load_mobilenetv2


def load_cnn_model(backbone, weights_path, input_shape):
    """
    Generic CNN loader.
    Currently supports only MobileNetV2.
    """

    if backbone == "mobilenetv2":
        return load_mobilenetv2(
            weights_path=weights_path,
            input_shape=input_shape,
        )

    raise ValueError(f"Unsupported backbone: {backbone}")
