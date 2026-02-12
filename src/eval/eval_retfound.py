import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.retfound_loader import load_retfound_model
from src.preprocessing.frequency import preprocess_frequency as np_preprocess_frequency


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
# Frequency pipeline (reuse repo preprocessing)
# ---------------------------------------------------------------------

def _freq_py(image_tensor):
    """
    NumPy/OpenCV preprocessing wrapped for tf.py_function.

    image_tensor: EagerTensor with shape (H, W, 3), dtype uint8 or float32
    returns: np.float32 array shape (224,224,3) in [0,1]
    """
    img = image_tensor.numpy()
    # ensure uint8-like range if decode produced uint8 (usually it does)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # This function already does:
    # - center_crop 800x800
    # - resize to target_size
    # - FFT magnitude clipped and normalized to 0..255 per channel
    img_freq = np_preprocess_frequency(img, target_size=(224, 224)).astype(np.float32)

    # match main_finetune_fourier.py scaling: Rescaling(1./255)
    img_freq = img_freq / 255.0
    return img_freq


def preprocess_frequency_tf(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Use tf.py_function to reuse NumPy/OpenCV FFT preprocessing
    image = tf.py_function(func=_freq_py, inp=[image], Tout=tf.float32)
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

    # mimic label_mode="categorical"
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

    # match main_finetune.py evaluation: softmax over logits, take class-1 prob
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]

    # recover y_true from categorical labels
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
