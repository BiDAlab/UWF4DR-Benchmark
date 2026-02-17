# Dataset Split Preparation

This directory contains the scripts used to reconstruct the exact experimental
train, validation, and test splits employed in the associated paper:

**Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**

For detailed information about the dataset origin, licensing constraints, and
split definitions, please refer to:

- `data/README.md`
- `docs/reproducibility.md`

---

## Overview

The split definition files provided under `data/splits/` define the fixed
partitioning protocol used in the paper (i.e., which image identifier belongs
to train, validation, or test).

The scripts in this directory:

- combine those split definition files,
- read the official UWF4DR ground-truth CSV files distributed with the dataset,
- resolve correct local image paths from the downloaded dataset folders,
- and generate ready-to-use CSV files for model training and evaluation.

No image data or medical annotations are redistributed.

---

## Requirements

These scripts require only a minimal Python environment with `pandas` installed.

Example installation:

```bash
pip install pandas
```

---

## Available Scripts

One script is provided per task:

- **Task 1 - Image Quality Assessment**: `prepare_split_task_1.py` 

- **Task 2 - Referable Diabetic Retinopathy (RDR) Identification**: `prepare_split_task_2.py` 

- **Task 3 - Diabetic Macular Edema (DME) Identification**: `prepare_split_task_3.py` 

Each script applies the corresponding split definition file from `data/splits/`
and resolves the correct image paths from the locally available UWF4DR dataset
folders.

---

## Usage

The scripts should be executed from the root of the repository using module execution:

```bash
python -m src.prepare_splits.prepare_split_task_<task_number> \
    --output_dir <OUTPUT_DIRECTORY> \
    [additional task-specific arguments]
```

To inspect the full list of arguments for a specific task:

```bash
python -m src.prepare_splits.prepare_split_task_<task_number> -h
```

Each task may require dataset-specific arguments (e.g., paths to the downloaded
UWF4DR dataset folders and official ground-truth CSV files).

---

## Output

Each script writes the prepared split files to the directory specified by the
`--output_dir` argument (e.g., `prepared_taskX/`), which will contain:

~~~
prepared_taskX/
├── train.csv
├── validation.csv
└── test.csv
~~~

Each generated CSV file contains:

- `image_path`: absolute path to the image on the local system

- `label`: binary task label

These files can be passed directly to the evaluation scripts in src/eval/
via the --splits_dir argument.

---

## Notes

- No image data is copied or redistributed.
- The split definition CSV files in data/splits/ do not contain labels. Labels are retrieved locally from the official UWF4DR ground-truth files.
- The reconstructed splits are fully deterministic and reproduce the exact
dataset partitions reported in the paper.

---
