# src/eval/eval_vit.py
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.vit_loader import load_vit_model, preprocess_input_vit
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


def get_vit_preprocess_fn(domain: str):
    """
    ViT preprocessing MUST match training:
      - Spatial: center crop 800x800 + resize 448 + color normalization
      - Frequency: center crop 800x800 + resize 448 + FFT magnitude (clipped)
      - Then: scale to [-1, 1] using (x/127.5)-1
    """
    if domain == "spatial":

        def preprocess(img_rgb: np.ndarray, target_size):
            x = preprocess_spatial(img_rgb, target_size)  # float32 in [0..255] approx
            x = preprocess_input_vit(tf.convert_to_tensor(x)).numpy()
            return x

        return preprocess

    if domain == "frequency":

        def preprocess(img_rgb: np.ndarray, target_size):
            x = preprocess_frequency(img_rgb, target_size)  # float32 in [0..255] approx
            x = preprocess_input_vit(tf.convert_to_tensor(x)).numpy()
            return x

        return preprocess

    raise ValueError(f"Unsupported domain: {domain}")


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
        help='Path to trained model (.keras) OR "none" to run with a dummy model',
    )
    parser.add_argument(
        "--splits_dir",
        required=True,
        help="Directory containing test.csv",
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # Task-specific settings (ViT uses 448 for ALL tasks)
    # -------------------------------------------------
    input_size = (448, 448)

    # -------------------------------------------------
    # Domain-specific preprocessing (ViT-specific)
    # -------------------------------------------------
    preprocess_fn = get_vit_preprocess_fn(domain=args.domain)

    # -------------------------------------------------
    # Load data
    # -------------------------------------------------
    print("📂 Loading test split")
    image_paths, y_true = load_test_split(args.splits_dir)
    print(f"Test samples: {len(image_paths)}")

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    model_path = args.model
    if model_path.strip().lower() in {"none", "null"}:
        model_path = None

    print(f"📦 Loading ViT-B/16 model from: {model_path if model_path else '(dummy model)'}")
    model = load_vit_model(
        weights_path=model_path,
        input_shape=(*input_size, 3),
        image_size=448,
        task=args.task,  # so L2 matches task3 vs task1/2 if building dummy or weights-only
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

        x = preprocess_fn(img, target_size=input_size)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)
        y_score.append(float(pred[0, 0]))

    y_score = np.array(y_score, dtype=np.float32)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    metrics = compute_metrics(y_true, y_score)

    print("\n📊 Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
