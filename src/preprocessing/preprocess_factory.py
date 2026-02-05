from src.preprocessing.spatial import preprocess_spatial
from src.preprocessing.frequency import preprocess_frequency

# MobileNetV2 preprocessing (Keras)
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

# ResNet18 preprocessing (classification-models-keras)
from classification_models.tfkeras import Classifiers
_, resnet18_preprocess = Classifiers.get("resnet18")


def get_preprocess_fn(domain, backbone):
    """
    Returns a preprocessing function adapted to:
    - domain: spatial | frequency
    - backbone: mobilenetv2 | resnet18
    """

    if domain == "spatial":

        if backbone == "mobilenetv2":
            imagenet_preprocess = mobilenet_preprocess

        elif backbone == "resnet18":
            imagenet_preprocess = resnet18_preprocess

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        def preprocess(img, target_size):
            # Crop + resize + color normalization
            img = preprocess_spatial(img, target_size)

            # Backbone-specific ImageNet preprocessing
            img = imagenet_preprocess(img)

            return img

        return preprocess

    elif domain == "frequency":
        # Frequency domain does NOT use ImageNet preprocessing
        def preprocess(img, target_size):
            return preprocess_frequency(img, target_size)

        return preprocess

    else:
        raise ValueError(f"Unsupported domain: {domain}")
