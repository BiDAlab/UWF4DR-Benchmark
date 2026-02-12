import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.retfound_loader import load_retfound_model
from src.preprocessing.frequency import center_crop, freq_transform_mag_clipped  # reutiliza EXACTO tu código


# ---------------------------------------------------------------------
# Spatial (RGB) pipeline (match main_finetune.py)
# ---------------------------------------------------------------------

def center_crop_and_resize(image, crop_size=(800, 800), resize_size=(224, 224)):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    offset_height = (height - crop_size[0]) // 2
    offset_width = (width - crop_size[1]) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height,
        offset_width,
        crop_size[0],
        crop_size[1],
    )
    image = tf.image.resize(image, resize_size)
    return image


def preprocess_spatial_tf(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # EXACTLY like main_finetune.py: first resize to (800, 1016)
    image = tf.image.resize(image, (800, 1016))

    # then center-crop 800x800 and resize to 224x224
    image = center_crop_and_resize(image, crop_size=(800, 800), resize_size=(224, 224))

    # normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# ---------------------------------------------------------------------
# Frequency pipeline (match how you generated & saved Fourier images)
# ---------------------------------------------------------------------

def _freq_from_path_py(path_tensor):
    """
    Replicate exactly the pipeline used to generate the saved Fourier images:
      cv2.imread (BGR) -> BGR2RGB -> center_crop(800) -> resize(224) ->
      FFT mag clipped p99 + cv2.normalize to 0..255 -> uint8 -> /255

    Returns float32 (224,224,3) in [0,1].
    """
    import cv2  # ensure available in env
    path = path_tensor.numpy().decode("utf-8")

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image with cv2.imread: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # EXACT as your offline generator
    img = center_crop(img, crop_size=(800, 800))
    img = cv2.resize(img, (224, 224))  # default interpolation is INTER_LINEAR

    mag = freq_transform_mag_clipped(img)  # returns float32 0..255 in your repo

    # Match offline saving to uint8 (quantization)
    mag_u8 = np.clip(np.rint(mag), 0, 255).astype(np.uint8)

    mag_u8 = mag_u8[..., ::-1]

    # Match main_finetune_fourier.py: Rescaling(1./255)
    out = mag_u8.astype(np.float32) / 255.0
    return out


def preprocess_frequency_tf(path, label):
    # Only use path -> cv2.imread inside py_function (do NOT decode with TF)
    image = tf.py_function(func=_freq_from_path_py, inp=[path], Tout=tf.float32)
    image.set_shape((224, 224, 3))
    return image, label


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def classification_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    best_idx = np.argmax(tpr - fpr)

    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]

    return dict(
        auroc=auroc,
        auprc=auprc,
        sensitivity=sensitivity,
        specificity=specificity,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RETFound model (splits_dir + domain)")
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--domain", required=True, choices=["spatial", "frequency"])
    parser.add_argument("--model", required=True, help="Path to fine-tuned .h5 weights")
    parser.add_argument("--splits_dir", required=True, help="Folder containing test.csv")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    test_csv = os.path.join(args.splits_dir, "test.csv")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found in: {args.splits_dir}")

    data = np.genfromtxt(test_csv, delimiter=",", dtype=str, skip_header=1)
    image_paths = data[:, 0]
    labels_int = data[:, 1].astype(int)

    labels = tf.keras.utils.to_categorical(labels_int, num_classes=args.num_classes).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if args.domain == "spatial":
        ds = ds.map(preprocess_spatial_tf, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_frequency_tf, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    print("Loading RETFound model...")
    model = load_retfound_model(args.model, num_classes=args.num_classes)

    print("Running inference...")
    logits = model.predict(ds, verbose=1)

    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]

    y_true = np.argmax(np.concatenate([y.numpy() for _, y in ds], axis=0), axis=1)

    print("Computing metrics...")
    metrics = classification_metrics(y_true, probs)

    print("\n=== RETFound Test Results ===")
    print(f"Task:         {args.task}")
    print(f"Domain:       {args.domain}")
    print(f"AUROC:        {metrics['auroc']:.6f}")
    print(f"AUPRC:        {metrics['auprc']:.6f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.6f}")
    print(f"Specificity:  {metrics['specificity']:.6f}")


if __name__ == "__main__":
    main()
