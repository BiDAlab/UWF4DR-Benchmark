import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.retfound_loader import load_retfound_model
from src.preprocessing.frequency import compute_dft  # reuse existing DFT


# ---------------------------------------------------------------------
# Spatial (RGB) pipeline
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
        crop_size[1]
    )

    image = tf.image.resize(image, resize_size)
    return image


def preprocess_spatial(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # EXACT pipeline from main_finetune.py
    image = tf.image.resize(image, (800, 1016))
    image = center_crop_and_resize(image)

    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# ---------------------------------------------------------------------
# Frequency pipeline
# ---------------------------------------------------------------------

def preprocess_frequency(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize to 800x800 before DFT (same spatial region as RGB)
    image = tf.image.resize(image, (800, 800))

    # Convert to numpy for DFT if needed
    image_np = image.numpy()
    image_np = compute_dft(image_np)  # existing DFT from frequency.py

    image = tf.convert_to_tensor(image_np, dtype=tf.float32)

    # Final resize to 224x224 (RETFound input size)
    image = tf.image.resize(image, (224, 224))

    image = image / 255.0
    return image, label


# Wrapper to allow numpy operations inside tf.data
def tf_preprocess_frequency(path, label):
    image, label = tf.py_function(
        func=preprocess_frequency,
        inp=[path, label],
        Tout=[tf.float32, tf.float32]
    )
    image.set_shape((224, 224, 3))
    label.set_shape((2,))
    return image, label


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def classification_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
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
    parser = argparse.ArgumentParser(description="Evaluate RETFound model")
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--domain", required=True,
                        choices=["spatial", "frequency"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    test_csv = os.path.join(args.splits_dir, "test.csv")
    data = np.genfromtxt(test_csv, delimiter=",", dtype=str, skip_header=1)

    image_paths = data[:, 0]
    labels = data[:, 1].astype(int)
    labels = tf.keras.utils.to_categorical(labels, num_classes=args.num_classes)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if args.domain == "spatial":
        ds = ds.map(preprocess_spatial, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(tf_preprocess_frequency, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    print("Loading RETFound model...")
    model = load_retfound_model(
        weights_path=args.model,
        num_classes=args.num_classes
    )

    print("Running inference...")
    logits = model.predict(ds, verbose=1)

    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]
    y_true = np.argmax(np.concatenate([y.numpy() for _, y in ds]), axis=1)

    print("Computing metrics...")
    metrics = classification_metrics(y_true, probs)

    print("\n=== RETFound Test Results ===")
    print(f"Domain:       {args.domain}")
    print(f"AUROC:        {metrics['auroc']:.6f}")
    print(f"AUPRC:        {metrics['auprc']:.6f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.6f}")
    print(f"Specificity:  {metrics['specificity']:.6f}")


if __name__ == "__main__":
    main()
