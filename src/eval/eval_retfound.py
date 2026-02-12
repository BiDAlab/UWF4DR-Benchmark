import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.retfound_loader import load_retfound_model


# ---------------------------------------------------------------------
# Preprocessing (exact reproduction of main_finetune.py)
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


def preprocess(path, label):
    # Read image
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # First resize EXACTLY like image_dataset_from_directory
    image = tf.image.resize(image, (800, 1016))

    # Center crop + final resize
    image = center_crop_and_resize(image)

    # Normalize
    image = tf.cast(image, tf.float32) / 255.0

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
    parser = argparse.ArgumentParser(description="Evaluate RETFound model (reproducible)")
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--model", required=True,
                        help="Path to fine-tuned .h5 model")
    parser.add_argument("--splits_dir", required=True,
                        help="Directory containing test.csv")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load test split
    # -----------------------------------------------------------------

    test_csv = os.path.join(args.splits_dir, "test.csv")

    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found in {args.splits_dir}")

    data = np.genfromtxt(test_csv, delimiter=",", dtype=str, skip_header=1)

    image_paths = data[:, 0]
    labels = data[:, 1].astype(int)

    # Convert labels to categorical (like label_mode="categorical")
    labels = tf.keras.utils.to_categorical(labels, num_classes=args.num_classes)

    # -----------------------------------------------------------------
    # Build dataset
    # -----------------------------------------------------------------

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    print("Loading RETFound model...")
    model = load_retfound_model(
        weights_path=args.model,
        num_classes=args.num_classes
    )

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    print("Running inference...")
    logits = model.predict(ds, verbose=1)

    # EXACTLY like main_finetune.py
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]

    # True labels
    y_true = np.argmax(np.concatenate([y.numpy() for _, y in ds]), axis=1)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    print("Computing metrics...")
    metrics = classification_metrics(y_true, probs)

    print("\n=== RETFound Test Results (Reproducible Mode) ===")
    print(f"AUROC:       {metrics['auroc']:.6f}")
    print(f"AUPRC:       {metrics['auprc']:.6f}")
    print(f"Sensitivity: {metrics['sensitivity']:.6f}")
    print(f"Specificity: {metrics['specificity']:.6f}")


if __name__ == "__main__":
    main()
