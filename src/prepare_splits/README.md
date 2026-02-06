# Dataset Split Preparation

This directory contains the scripts used to reconstruct the exact experimental
train, validation, and test splits employed in the associated paper.

The splits are **independently defined by the authors** and do not correspond to
the official (hidden) test split of the UWF4DR Challenge. They are designed to
enable full reproducibility of the evaluation protocol outside the competition
platform.

---

## Purpose

The scripts in this directory allow users to:

- reconstruct the same experimental splits used in the paper,
- starting from the officially downloaded UWF4DR datasets,
- while respecting dataset licensing and annotation redistribution constraints.

Each script generates CSV files that explicitly define which images belong to
the training, validation, and test subsets.

These CSV files constitute the **sole reference definition** of the experimental
splits used throughout the repository.

---

## Available Scripts

One script is provided per task:

- **Task 1 – Image Quality Assessment**  
  `prepare_split_task_1.py`

- **Task 2 – Referable Diabetic Retinopathy (RDR) Identification**  
  `prepare_split_task_2.py`

- **Task 3 – Diabetic Macular Edema (DME) Identification**  
  `prepare_split_task_3.py`

Each script applies the same fixed split definition reported in the paper and
resolves the correct image paths from the locally available UWF4DR dataset
folders.

---

## Output

When executed successfully, each script produces the following files:

train.csv
validation.csv
test.csv


Each CSV file contains:

- the absolute path to each image on the local system,
- the corresponding binary label for the task.

The generated CSV files are directly consumed by the evaluation scripts in
`src/eval/`.

---

## Execution Notes

- The scripts are intended to be executed from the **root of the repository**,
  preferably using Python module execution (e.g., `python -m`).
- No image data or official annotations are copied or redistributed.
- The scripts only generate split definitions and resolved image paths.

For a detailed description of the experimental protocol and dataset handling,
refer to `docs/reproducibility.md`.
