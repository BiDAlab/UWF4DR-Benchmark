import os
import argparse
import pandas as pd


# Fixed path to the split definition used in the paper
TASK1_SPLIT_CSV = "data/splits/task1_split.csv"


def load_official_annotations(task1_training_root, task1_validation_root):
    """
    Load official Task 1 annotations from the UWF4DR dataset.

    Training and Validation are TWO different directories
    even though they have the same name when downloaded.
    """

    records = []

    train_gt = os.path.join(
        task1_training_root, "2. Groundtruths", "1. Training.csv"
    )
    if not os.path.isfile(train_gt):
        raise FileNotFoundError(f"Training groundtruth not found: {train_gt}")

    df_train = pd.read_csv(train_gt)
    if "image" not in df_train.columns:
        raise RuntimeError("'image' column not found in Training groundtruth CSV")

    records.append(df_train)

    val_gt = os.path.join(
        task1_validation_root, "2. Groundtruths", "2. Validation.csv"
    )
    if not os.path.isfile(val_gt):
        raise FileNotFoundError(f"Validation groundtruth not found: {val_gt}")

    df_val = pd.read_csv(val_gt)
    if "image" not in df_val.columns:
        raise RuntimeError("'image' column not found in Validation groundtruth CSV")

    records.append(df_val)

    return pd.concat(records, ignore_index=True)


def find_image(image_id, search_roots):
    """
    Search for an image file across multiple dataset roots.

    Each root must follow the official UWF4DR structure.
    """

    for root in search_roots:
        for subfolder in [("1. Images", "1. Training"),
                          ("1. Images", "2. Validation")]:
            candidate = os.path.join(root, *subfolder, image_id)
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(
        f"Image '{image_id}' not found in any provided dataset roots."
    )


def build_task1_datasets(
    task1_training_root,
    task1_validation_root,
    task23_training_root,
    task23_validation_root
):
    """
    Build train/validation/test datasets for Task 1 using:
    - fixed split definition from the repository
    - official Task 1 annotations
    - image files possibly located in Task 1 or Task 2/3 folders
    """

    split_df = pd.read_csv(TASK1_SPLIT_CSV)
    annotations = load_official_annotations(
        task1_training_root, task1_validation_root
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
            f"Missing labels for {len(missing)} images "
            f"(first 5 shown): {missing['image_id'].tolist()[:5]}"
        )

    label_cols = [c for c in merged.columns if c not in ("image_id", "split")]
    if len(label_cols) != 1:
        raise RuntimeError(
            f"Expected exactly one label column, found {label_cols}"
        )
    label_col = label_cols[0]

    search_roots = [
        task1_training_root,
        task1_validation_root,
        task23_training_root,
        task23_validation_root,
    ]

    datasets = {"train": [], "validation": [], "test": []}

    for _, row in merged.iterrows():
        image_path = find_image(row["image_id"], search_roots)
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
            "IMPORTANT: Training and Validation sets must be provided as "
            "separate directories, exactly as downloaded from the competition."
        )
    )

    parser.add_argument("--task1_training_root", required=True)
    parser.add_argument("--task1_validation_root", required=True)
    parser.add_argument("--task23_training_root", required=True)
    parser.add_argument("--task23_validation_root", required=True)

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory to save the prepared datasets as CSV files"
    )

    args = parser.parse_args()

    datasets = build_task1_datasets(
        args.task1_training_root,
        args.task1_validation_root,
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

