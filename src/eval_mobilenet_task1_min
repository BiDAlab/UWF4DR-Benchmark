import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Needed when generating splits in memory
from prepare_split_task_1 import build_task1_datasets


# =================================================
# Image preprocessing (same as training)
# =================================================

def center_crop(image, crop_size=(800, 800)):
    h, w, _ = image.shape
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return image[top:top + ch, left:left + cw]


def color_normalization(image, blur_radius=9, amplification_factor=4, offset=128):
    r, g, b = image.split()

    r_blur = r.filter(ImageFilter.GaussianBlur(blur_radius))
    g_blur = g.filter(ImageFilter.GaussianBlur(blur_radius))
    b_blur = b.filter(ImageFilter.GaussianBlur(blur_radius))

    def normalize(c, c_blur):
        c = np.asarray(c, np.float32)
        c_blur = np.asarray(c_blur, np.float32)
        c = c - c_blur
        c = np.clip(c * amplification_factor + offset, 0, 255)
        return Image.fromarray(c.astype(np.uint8), mode="L")

    r = normalize(r, r_blur)
    g = normalize(g, g_blur)
    b = normalize(b, b_blur)

    return Image.merge("RGB", (r, g, b))


def _preprocess_numpy(img):
    img = Image.fromarray(img)
    img = np.array(img)

    img = center_crop(img, (800, 800))
    img = tf.image.resize(img, (448, 448)).numpy().astype(np.uint8)

    img = Image.fromarray(img)
    img = color_normalization(img)

    img = np.asarray(img)
    img = preprocess_input(img)

    return img.astype(np.float32)


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.numpy_function(
        _preprocess_numpy,
        [img],
        Tout=tf.float32
    )
    img.set_shape((448, 448, 3))

    return img, tf.cast(label, tf.float32)


# =================================================
# Metrics
# =================================================

def compute_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = np.argmax(tpr - fpr)

    sensitivity = tpr[best_idx]
    specificity = 1.0 - fpr[best_idx]

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


# =================================================
# Main
# =================================================

def main():
    parser = argparse.ArgumentParser(
        description="Minimal evaluation script for Task 1 (MobileNetV2)."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.keras or .h5)"
    )

    # Option A: CSV splits
    parser.add_argument("--splits_dir", default=None)

    # Option B: generate splits in memory
    parser.add_argument("--task1_training_root", default=None)
    parser.add_argument("--task1_validation_root", default=None)
    parser.add_argument("--task23_training_root", default=None)
    parser.add_argument("--task23_validation_root", default=None)

    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    # -------------------------------------------------
    # Load test split
    # -------------------------------------------------

    if args.splits_dir is not None:
        print("📂 Loading test split from CSV")
        test_df = pd.read_csv(os.path.join(args.splits_dir, "test.csv"))
        test_paths = test_df["image_path"].tolist()
        test_labels = test_df["label"].tolist()

    else:
        print("🧠 Generating dataset splits in memory")
        datasets = build_task1_datasets(
            args.task1_training_root,
            args.task1_validation_root,
            args.task23_training_root,
            args.task23_validation_root,
        )
        test_paths, test_labels = zip(*datasets["test"])
        test_paths = list(test_paths)
        test_labels = list(test_labels)

    print(f"Test samples: {len(test_paths)}")

    # -------------------------------------------------
    # tf.data pipeline
    # -------------------------------------------------

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = (
        test_ds
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # -------------------------------------------------
    # Load model and evaluate
    # -------------------------------------------------

    print(f"📦 Loading model from: {args.model}")
    model = load_model(args.model)

    y_true = []
    y_score = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_score.extend(preds.flatten())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    metrics = compute_metrics(y_true, y_score)

    print("\n📊 Evaluation results (Task 1):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
