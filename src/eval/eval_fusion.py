import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.models.fusion_loader import load_fusion_model


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_features(csv_path):
    """
    Load feature CSV.
    Assumes:
        - All columns except 'label' are feature dimensions
        - Column 'label' exists
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {csv_path}")

    labels = df["label"].values
    features = df.drop(columns=["label"]).values

    return features, labels


def compute_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = np.argmax(tpr - fpr)

    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]

    return auroc, auprc, sensitivity, specificity


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3"])
    parser.add_argument("--domain", required=True,
                        choices=["spatial", "frequency"])
    parser.add_argument("--features_root", required=True,
                        help="Path to data/fusion")
    parser.add_argument("--mlp_model", required=True,
                        help="Path to fusion MLP (.keras)")

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Construct feature directory
    # -----------------------------------------------------------------

    features_dir = os.path.join(
        args.features_root,
        args.domain,
        args.task
    )

    if not os.path.isdir(features_dir):
        raise ValueError(f"Features directory not found: {features_dir}")

    # Model names (fixed order for concatenation)
    model_names = [
        "mobilenetv2",
        "resnet18",
        "vitb16",
        "retfound"
    ]

    train_features_list = []
    test_features_list = []

    # -----------------------------------------------------------------
    # Load train/test features and fit scalers
    # -----------------------------------------------------------------

    for model_name in model_names:

        train_csv = os.path.join(
            features_dir,
            f"features_{model_name}_{args.task}_train.csv"
        )

        test_csv = os.path.join(
            features_dir,
            f"features_{model_name}_{args.task}_test.csv"
        )

        if not os.path.exists(train_csv):
            raise FileNotFoundError(train_csv)

        if not os.path.exists(test_csv):
            raise FileNotFoundError(test_csv)

        # Load
        X_train, y_train = load_features(train_csv)
        X_test, y_test = load_features(test_csv)

        # Fit scaler ONLY on train
        scaler = StandardScaler().fit(X_train)

        X_test_scaled = scaler.transform(X_test)

        train_features_list.append(X_train)
        test_features_list.append(X_test_scaled)

    # -----------------------------------------------------------------
    # Concatenate (fixed order)
    # -----------------------------------------------------------------

    X_test = np.concatenate(test_features_list, axis=1)

    # Labels from last loaded model (all identical)
    y_true = y_test

    # -----------------------------------------------------------------
    # Load fusion MLP
    # -----------------------------------------------------------------

    fusion_model = load_fusion_model(args.mlp_model)

    y_score = fusion_model.predict(X_test, verbose=0).flatten()

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    auroc, auprc, sensitivity, specificity = compute_metrics(
        y_true,
        y_score
    )

    print(f"\n=== Fusion Evaluation ({args.task} | {args.domain}) ===")
    print(f"AUROC:       {auroc:.4f}")
    print(f"AUPRC:       {auprc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("")


if __name__ == "__main__":
    main()
