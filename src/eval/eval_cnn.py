import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.mobilenetv2 import load_mobilenetv2
from src.preprocessing.spatial import preprocess_spatial
from src.preprocessing.frequency import preprocess_frequency


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def load_test_split(splits_dir):
    test_csv = os.path.join(splits_dir, "test.csv")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test split not found: {test_csv}")

    data = np.genfromtxt(test_csv, delimiter=",", dtype=str, skip_header=1)
    image_paths = data[:, 0]
    labels = data[:, 1].astype(int)

    return image_paths, labels


def compute_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)

    sensitivity = tpr[idx]
    specificity = 1 - fpr[idx]

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2 models on the UWF4DR dataset"
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=["task1", "task2", "task3"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["spatial", "frequency"],
        help="Input domain",
    )
    parser.add_argument(
    "--backbone",
    type=str,
    default="mobilenetv2",
    choices=["mobilenetv2", "resnet18"],
    help="CNN backbone architecture"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.keras)",
    )
    parser.add_argument(
        "--splits_dir",
        required=True,
        help="Directory containing test.csv",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Task-specific settings
    # -------------------------------------------------

    if args.task == "task1":
        input_size = (448, 448)
    else:
        input_size = (800, 800)

    # -------------------------------------------------
    # Domain-specific preprocessing
    # -------------------------------------------------

    if args.domain == "spatial":
        preprocess_fn = preprocess_spatial
    else:
        preprocess_fn = preprocess_frequency

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    print("📂 Loading test split")
    image_paths, y_true = load_test_split(args.splits_dir)
    print(f"Test samples: {len(image_paths)}")

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------

    print(f"📦 Loading model from: {args.model}")
    model = load_mobilenetv2(
        weights_path=args.model,
        input_shape=(*input_size, 3),
    )

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------

    y_score = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = preprocess_fn(img, target_size=input_size)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img, verbose=0)
        y_score.append(pred[0, 0])

    y_score = np.array(y_score)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    metrics = compute_metrics(y_true, y_score)

    print("\n📊 Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
