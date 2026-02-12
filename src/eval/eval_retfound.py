import argparse
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.retfound_loader import load_retfound_model


# ---------------------------------------------------------------------
# Preprocessing (RETFound-specific)
# ---------------------------------------------------------------------

def center_crop(image, crop_size=(800, 800)):
    h, w, _ = image.shape
    new_w, new_h = crop_size

    left = (w - new_w) // 2
    top = (h - new_h) // 2

    return image[top:top+new_h, left:left+new_w]


def preprocess_retfound(img):
    """
    Preprocessing pipeline used for RETFound:

    - center crop 800x800
    - resize to 224x224
    - normalize to [0,1]
    """
    img = center_crop(img, crop_size=(800, 800))
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    return img


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_split_csv(splits_dir):
    csv_path = os.path.join(splits_dir, "test.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"test.csv not found in {splits_dir}")

    data = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1)
    image_paths = data[:, 0]
    labels = data[:, 1].astype(int)

    return image_paths, labels


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
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--model", required=True, help="Path to fine-tuned .h5 model")
    parser.add_argument("--splits_dir", required=True, help="Directory containing test.csv")
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    print("Loading test split...")
    image_paths, y_true = load_split_csv(args.splits_dir)
    print(f"Number of test samples: {len(image_paths)}")

    print("Loading RETFound model...")
    model = load_retfound_model(
        weights_path=args.model,
        num_classes=args.num_classes
    )

    print("Running inference...")
    y_scores = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_retfound(img)

        x = np.expand_dims(img, axis=0)
        pred = model.predict(x, verbose=0)

        # If model outputs 2 logits, take probability of class 1
        if pred.shape[-1] == 2:
            score = pred[0, 1]
        else:
            score = pred.flatten()[0]

        y_scores.append(score)

    y_scores = np.array(y_scores)

    print("Computing metrics...")
    metrics = classification_metrics(y_true, y_scores)

    print("\n=== RETFound Test Results ===")
    print(f"AUROC:       {metrics['auroc']:.4f}")
    print(f"AUPRC:       {metrics['auprc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")


if __name__ == "__main__":
    main()
