# Model Evaluation

This directory contains the evaluation scripts used to assess pretrained deep learning models on the **fixed experimental test splits** defined in the paper.

The evaluation code:

- loads `test.csv` produced by `src/prepare_splits/`,
- resolves local image paths,
- applies the appropriate preprocessing pipeline (spatial or frequency),
- loads a pretrained model,
- runs inference on the test set,
- computes standardized binary classification metrics.

---

## Scope

- ✅ Reproducible evaluation on fixed splits (train/val/test already defined by the authors’ split CSVs).
- ✅ Deterministic preprocessing at evaluation time.
- ❌ No training, fine-tuning, or hyperparameter search.
- ❌ No dataset hosting and no model weights hosted in this repository.

---

## Prerequisites

### 1) Dataset available locally
You must download the official **UWF4DR** dataset and keep its original folder structure.

### 2) Prepared split CSVs (required)
Before running any evaluation, generate the split CSVs using the corresponding script in:

- `src/prepare_splits/prepare_split_task_1.py`
- `src/prepare_splits/prepare_split_task_2.py`
- `src/prepare_splits/prepare_split_task_3.py`

Each script outputs an `output_dir/` containing:
- `train.csv`
- `validation.csv`
- `test.csv`

The evaluation scripts expect **`test.csv`** inside the provided `--splits_dir`.

### 3) Python environment
Install the dependencies required for the model family you want to evaluate (see `requirements/`).

### 4) Pretrained model files
Provide local paths to the pretrained model files when running the scripts (see each script below).

---

## Available Evaluation Scripts

### 1) CNN evaluation (`eval_cnn.py`)

Evaluates CNN backbones:

- `mobilenetv2`
- `resnet18`

**Usage**

```bash
python -m src.eval.eval_cnn \
  --task {task1,task2,task3} \
  --domain {spatial,frequency} \
  --backbone {mobilenetv2,resnet18} \
  --model /path/to/cnn_model.keras \
  --splits_dir /path/to/prepared_taskX
```

**Arguments**

- `--task` (required): `task1`, `task2`, `task3`
- `--domain` (required): `spatial` or `frequency`
- `--backbone` (optional): `mobilenetv2` (default) or `resnet18`
- `--model` (required): path to a pretrained model file
- `--splits_dir` (required): directory containing `test.csv`

---

### 2) ViT evaluation (`eval_vit.py`)

Evaluates:

- `ViT-B/16`

**Usage**

```bash
python -m src.eval.eval_vit \
  --task {task1,task2,task3} \
  --domain {spatial,frequency} \
  --model /path/to/vit_model.keras \
  --splits_dir /path/to/prepared_taskX
```

**Arguments**

- `--task` (required): `task1`, `task2`, `task3`
- `--domain` (required): `spatial` or `frequency`
- `--model` (required): path to the pretrained ViT model
- `--splits_dir` (required): directory containing `test.csv`

---

### 3) RETFound evaluation (`eval_retfound.py`)

Evaluates:

- `RETFound` (retinal foundation model)

**Usage**

```bash
python -m src.eval.eval_retfound \
  --task {task1,task2,task3} \
  --domain {spatial,frequency} \
  --model /path/to/retfound_weights.h5 \
  --splits_dir /path/to/prepared_taskX
```

**Arguments**

- `--task` (required): `task1`, `task2`, `task3`
- `--domain` (required): `spatial` or `frequency`
- `--model` (required): path to the pretrained RETFound weights file
- `--splits_dir` (required): directory containing `test.csv`

**Optional arguments**

- `--num_classes` (default: `2`)
- `--batch_size` (default: `32`)

**Note:** RETFound evaluation uses a dedicated preprocessing path within this script to match the original configuration used for the model.

---

### 4) Fusion evaluation (`eval_fusion.py`)

Evaluates the feature-level fusion MLP using pre-extracted feature CSVs.

This script expects feature files under:

`<features_root>/<domain>/<task>/`

and, for each backbone (fixed order: `mobilenetv2`, `resnet18`, `vitb16`, `retfound`), the following files:

- `features_<model>_<task_#>_train.csv`
- `features_<model>_<task_#>_test.csv`

Each features CSV must contain:

- feature columns (any number)
- a `label` column

**Usage**

```bash
python -m src.eval.eval_fusion \
  --task {task1,task2,task3} \
  --domain {spatial,frequency} \
  --features_root /path/to/data/fusion \
  --mlp_model /path/to/fusion_mlp.keras
```

**Arguments**

- `--task` (required): `task1`, `task2`, `task3`
- `--domain` (required): `spatial` or `frequency`
- `--features_root` (required): root folder containing the fusion features
- `--mlp_model` (required): path to the fusion MLP model

---

## Metrics and Output

All evaluation scripts compute standard binary classification metrics:

- AUROC
- AUPRC
- Sensitivity
- Specificity

Metrics are printed to stdout. No intermediate predictions are saved unless the user modifies the scripts.

Threshold-dependent metrics are computed using an operating point derived from the ROC curve (maximizing `TPR - FPR`).

---

## Notes

- Evaluation is deterministic given the same `test.csv`, model file, and environment.
- Minor numeric differences may occur across hardware/software setups, but the evaluation protocol remains fixed.
- Run scripts from the repository root using `python -m` to ensure imports resolve correctly.

For the full experimental protocol and end-to-end workflow, see:

- `docs/reproducibility.md`

---
