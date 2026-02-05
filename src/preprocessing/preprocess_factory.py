from src.preprocessing.spatial import preprocess_spatial
from src.preprocessing.frequency import preprocess_frequency

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess


def get_preprocess_fn(domain, backbone):
    """
    Returns a preprocessing function adapted to the
    input domain (spatial / frequency) and CNN backbone.
    """

    if domain == "spatial":
        if backbone == "mobilenetv2":
            base_preprocess = mobilenet_preprocess
        elif backbone == "resnet18":
            base_preprocess = resnet_preprocess
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        def preprocess(img, target_size):
            img = preprocess_spatial(img, target_size)
            img = base_preprocess(img)
            return img

        return preprocess

    elif domain == "frequency":
        # Frequency domain does NOT use ImageNet preprocessing
        def preprocess(img, target_size):
            return preprocess_frequency(img, target_size)

        return preprocess

    else:
        raise ValueError(f"Unsupported domain: {domain}")
