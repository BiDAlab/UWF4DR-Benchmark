# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:53:20 2026

@author: pablo
"""

import os
import argparse
import pandas as pd


def load_official_annotations(task_root):
    records = []

    for split_name in ["Training", "Validation"]:
        # NOTE: Folder name must match the official dataset structure exactly
        gt_dir = os.path.join(task_root, split_name, "2. Groundtruths")

        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"Groundtruths folder not found: {gt_dir}")

        csv_files = [f for f in os.listdir(gt_dir) if f.endswith(".csv")]
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected 1 CSV in {gt_dir}, found {csv_files}")

        df = pd.read_csv(os.path.join(gt_dir, csv_files[0]))

        if "image" not in df.columns:
            raise RuntimeError("'image' column not found in groundtruth CSV")

        records.append(df)

    return pd.concat(records, ignore_index=True)


def get_image_path(task_root, image_id):
    for split_name in ["Training", "Validation"]:
        p = os.path.join(task_root, split_name, "1. Images", image_id)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Image not found: {image_id}")


def build_task1_datasets(task_root, split_csv):
    split_df = pd.read_csv(split_csv)
    annotations = load_official_annotations(task_root)

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
        img_path = get_image_path(task_root, row["image_id"])
        datasets[row["split"]].append((img_path, row[label_col]))

    return datasets


def save_datasets(datasets, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split, samples in datasets.items():
        df = pd.DataFrame(samples, columns=["image_path", "label"])
        df.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Task 1 dataset splits using official UWF4DR annotations"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--split_csv",
        default="data/splits/task1_split.csv"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save prepared datasets"
    )

    args = parser.parse_args()

    datasets = build_task1_datasets(args.data_root, args.split_csv)

    for split, samples in datasets.items():
        print(f"{split}: {len(samples)} samples")

    if args.output_dir:
        save_datasets(datasets, args.output_dir)
        print(f"📁 Datasets saved to: {args.output_dir}")


if __name__ == "__main__":

    main()
