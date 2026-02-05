from src.models.mobilenetv2 import load_mobilenetv2
from src.models.resnet18 import load_resnet18


def load_cnn_model(backbone, weights_path, input_shape):

    if backbone == "mobilenetv2":
        return load_mobilenetv2(
            weights_path=weights_path,
            input_shape=input_shape,
        )

    elif backbone == "resnet18":
        return load_resnet18(
            weights_path=weights_path,
            input_shape=input_shape,
        )

    raise ValueError(f"Unsupported backbone: {backbone}")
