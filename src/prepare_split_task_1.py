# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:53:20 2026

@author: pablo
"""

import os
import argparse
import pandas as pd


# Fixed path to the split definition used in the paper
TASK1_SPLIT_CSV = "data/splits/task1_split.csv"


def load_official_annotations(training_root, validation_root):
    """
    Load official Task 1 annotations from the UWF4DR dataset.

    IMPORTANT:
    - training_root and validation_root are TWO DIFFERENT directories,
      even though they have the same name when downloaded.
    """

    records = []

    # Training annotations
    train_gt = os.path.join(training_root, "2. Groundtruths", "1. Training.csv")
    if not os.path.isfile(train_gt):
        raise FileNotFoundError(f"Training groundtruth not found: {train_gt}")

    df_train = pd.read_csv(train_gt)
    if "image" not in df_train.columns:
        raise RuntimeError("'image' column not found in Training groundtruth CSV")

    records.append(df_train)

    # Validation annotations
    val_gt = os.path.join(validation_root, "2. Groundtruths", "2. Validation.csv")
    if not os.path.isfile(val_gt):
        raise FileNotFoundError(f"Validation groundtruth not found: {val_gt}")

    df_val = pd.read_csv(val_gt)
    if "image" not in df_val.columns:
        raise RuntimeError("'image' column not found in Validation groundtruth CSV")

    records.append(df_val)

    return pd.concat(records, ignore_index=True)


def get_image_path(training_root, validation_root, image_id):
    """
    Locate an image file either in the Training or Validation image folders.
    """

    train_img = os.path.join(
        training_root, "1. Images", "1. Training", image_id
    )
    if os.path.exists(train_img):
        return train_img

    val_img = os.path.join(
        validation_root, "1. Images", "2. Validation", image_id
    )
    if os.path.exists(val_img):
        return val_img

    raise FileNotFoundError(f"Image not found in Training or Validation: {image_id}")


def build_task1_datasets(training_root, validation_root):
    """
    Build train/validation/test datasets using:
    - the fixed split definition provided in the repository
    - the official UWF4DR Task 1 annotations
    """

    split_df = pd.read_csv(TASK1_SPLIT_CSV)
    annotations = load_official_annotations(training_root, validation_root)

    merged = split_df.merge(
        annotations,
        left_on="image_id",
        right_on="image",
        how="left"
    ).drop(columns=["image"])

    if merged.isna().any().any():
        missing = merged[merged.isna().any(axis=1)]
        raise RuntimeError(
            f"Missing labels for {len(missing)} images "
            f"(first 5 shown): {missing['image_id'].tolist()[:5]}"
        )

    label_cols = [c for c in merged.columns if c not in ("image_id", "split")]
    if len(label_cols) != 1:
        raise RuntimeError(
            f"Expected exactly one label column, found {label_cols}"
        )
    label_col = label_cols[0]

    datasets = {"train": [], "validation": [], "test": []}

    for _, row in merged.iterrows():
        image_path = get_image_path(
            training_root, validation_root, row["image_id"]
        )
        datasets[row["split"]].append((image_path, row[label_col]))

    return datasets


def save_datasets(datasets, output_dir):
    """
    Save prepared datasets as CSV files (image_path, label).
    """
    os.makedirs(output_dir, exist_ok=True)

    for split, samples in datasets.items():
        df = pd.DataFrame(samples, columns=["image_path", "label"])
        df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Task 1 dataset splits using official UWF4DR annotations.\n\n"
            "IMPORTANT: Training and Validation sets must be provided as TWO "
            "separate directories, as downloaded from the competition."
        )
    )
    parser.add_argument(
        "--training_root",
        required=True,
        help="Path to the downloaded Task 1 TRAINING set"
    )
    parser.add_argument(
        "--validation_root",
        required=True,
        help="Path to the downloaded Task 1 VALIDATION set"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save the prepared datasets as CSV files"
    )

    args = parser.parse_args()

    datasets = build_task1_datasets(
        args.training_root,
        args.validation_root
    )

    for split, samples in datasets.items():
        print(f"{split}: {len(samples)} samples")

    if args.output_dir:
        save_datasets(datasets, args.output_dir)
        print(f"📁 Datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
