import argparse
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.vit_loader import load_vit_model
from src.preprocessing.preprocess_factory import get_preprocess_fn


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
        description="Evaluate ViT-B/16 models on the UWF4DR dataset"
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
        "--model",
        required=True,
        help="Path to trained ViT model (.keras)",
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
    # IMPORTANT: backbone="vit" must be supported by preprocess_factory
    preprocess_fn = get_preprocess_fn(
        domain=args.domain,
        backbone="vit",
    )

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------

    print("📂 Loading test split")
    image_paths, y_true = load_test_split(args.splits_dir)
    print(f"Test samples: {len(image_paths)}")

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------

    print(f"📦 Loading ViT model from: {args.model}")
    model = load_vit_model(
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

        # If your frequency preprocessor expects RGB input, keep this conversion.
        # If it expects grayscale/BGR, it should handle conversion internally.
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
