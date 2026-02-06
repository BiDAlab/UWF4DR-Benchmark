# Dataset Split Preparation

This directory contains the scripts used to reconstruct the exact experimental
train, validation, and test splits employed in the associated paper.

The experimental split assignments (i.e., which image belongs to train,
validation, or test) are defined by the authors and are provided in the
repository under `data/splits/` as CSV files containing image identifiers
and their split assignment.

The scripts in this directory combine:

- the split definition files in `data/splits/`, and
- the official UWF4DR ground-truth CSV files distributed with the dataset,

to generate locally usable split files with resolved image paths and task labels.

---

## Purpose

The scripts in this directory allow users to:

- reproduce the same experimental splits used in the paper,
- starting from the officially downloaded UWF4DR dataset,
- while respecting dataset licensing and annotation redistribution constraints.

In particular, the scripts materialize the experimental protocol by producing
CSV files that can be consumed directly by the evaluation code.

---

## Available Scripts

One script is provided per task:

- **Task 1 – Image Quality Assessment**  
  `prepare_split_task_1.py`

- **Task 2 – Referable Diabetic Retinopathy (RDR) Identification**  
  `prepare_split_task_2.py`

- **Task 3 – Diabetic Macular Edema (DME) Identification**  
  `prepare_split_task_3.py`

Each script applies the corresponding split definition file from `data/splits/`
and resolves the correct image paths from the locally available UWF4DR dataset
folders.

---

## Output

Each script writes the prepared split files to the directory specified by the
`--output_dir` argument:

output_dir/
├── train.csv
├── validation.csv
└── test.csv


Each output CSV contains:

- `image_path`: absolute path to the image on the local system
- `label`: binary label for the corresponding task

These generated CSV files are the inputs expected by the evaluation scripts in
`src/eval/`.

---

## Execution Notes

- The scripts are intended to be executed from the **root of the repository**,
  preferably using Python module execution (e.g., `python -m`).
- No image data is copied or redistributed.
- The split assignment CSV files in `data/splits/` do not contain medical labels.
  Labels are obtained locally from the official UWF4DR ground-truth files.

For a detailed description of the experimental protocol and dataset handling,
refer to `docs/reproducibility.md`.
