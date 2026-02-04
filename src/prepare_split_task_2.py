import os
import argparse
import pandas as pd


# Fixed path to the split definition used in the paper
TASK2_SPLIT_CSV = "data/splits/task2_split.csv"


def load_official_annotations(task23_training_root, task23_validation_root):
    """
    Load official Task 2/3 annotations from the UWF4DR dataset.

    ALL images belong to Task 2 (RDR Identification).
    """

    records = []

    train_gt = os.path.join(
        task23_training_root, "2. Groundtruths", "1. Training.csv"
    )
    if not os.path.isfile(train_gt):
        raise FileNotFoundError(f"Training groundtruth not found: {train_gt}")

    records.append(pd.read_csv(train_gt))

    val_gt = os.path.join(
        task23_validation_root, "2. Groundtruths", "2. Validation.csv"
    )
    if not os.path.isfile(val_gt):
        raise FileNotFoundError(f"Validation groundtruth not found: {val_gt}")

    records.append(pd.read_csv(val_gt))

    df = pd.concat(records, ignore_index=True)

    if "image" not in df.columns:
        raise RuntimeError("'image' column not found in groundtruth CSV")

    return df


def find_image(image_id, search_roots):
    """
    Locate an image file in Task 2/3 image folders.
    """

    for root in search_roots:
        for subfolder in [("1. Images", "1. Training"),
                          ("1. Images", "2. Validation")]:
            candidate = os.path.join(root, *subfolder, image_id)
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        f"Image '{image_id}' not found in Task 2/3 image folders."
    )


def build_task2_datasets(task23_training_root, task23_validation_root):
    """
    Build train/validation/test datasets for Task 2 (RDR Identification)
    using:
    - fixed split definition from the repository
    - official Task 2/3 annotations
    """

    split_df = pd.read_csv(TASK2_SPLIT_CSV)
    annotations = load_official_annotations(
        task23_training_root, task23_validation_root
    )

    merged = split_df.merge(
        annotations,
        left_on="image_id",
        right_on="image",
        how="left"
    ).drop(columns=["image"])

    if merged.isna().any().any():
        missing = merged[merged.isna().any(axis=1)]
        raise RuntimeError(
            f"Missing annotations for {len(missing)} images "
            f"(first 5 shown): {missing['image_id'].tolist()[:5]}"
        )

    # Identify label column for Task 2 (RDR)
    label_cols = [c for c in merged.columns if c not in ("image_id", "split")]

    if len(label_cols) < 1:
        raise RuntimeError("No label columns found in annotations")

    rdr_col = label_cols[0]  # Task 2 label column

    datasets = {"train": [], "validation": [], "test": []}

    search_roots = [
        task23_training_root,
        task23_validation_root,
    ]

    for _, row in merged.iterrows():
        image_path = find_image(row["image_id"], search_roots)
        datasets[row["split"]].append((image_path, row[rdr_col]))

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
            "Prepare Task 2 (RDR Identification) dataset splits using "
            "official UWF4DR annotations."
        )
    )

    parser.add_argument("--task23_training_root", required=True)
    parser.add_argument("--task23_validation_root", required=True)

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save the prepared datasets as CSV files"
    )

    args = parser.parse_args()

    datasets = build_task2_datasets(
        args.task23_training_root,
        args.task23_validation_root
    )

    for split, samples in datasets.items():
        print(f"{split}: {len(samples)} samples")

    if args.output_dir:
        save_datasets(datasets, args.output_dir)
        print(f"📁 Datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
