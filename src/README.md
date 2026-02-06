# Source Code

This directory contains the core Python source code used to reproduce the
experimental protocol and evaluation pipeline described in the associated paper.

The code in `src/` is designed to support **reproducible evaluation** of deep
learning models based on fixed experimental splits defined by the authors and
deterministic preprocessing pipelines.

This directory does **not** provide a full end-to-end training framework.
Models are assumed to have been trained offline as described in the paper and
are evaluated here using standardized inputs and splits.

---

## Scope and Design Principles

The code in this directory follows these principles:

- **Reproducibility-first**:  
  All evaluations are driven by explicit CSV split definitions and deterministic
  preprocessing steps.

- **No data hosting**:  
  Raw images and official annotations are never stored or redistributed here.
  Image paths are resolved locally from the user’s downloaded UWF4DR dataset.

- **Separation of concerns**:  
  Dataset preparation, preprocessing, model definitions, and evaluation are
  clearly separated into dedicated submodules.

- **Model-agnostic evaluation**:  
  Evaluation scripts are designed to work with different backbone architectures,
  as long as compatible model weights are provided.

---

## Directory Overview

The `src/` directory is organized as follows:

- `prepare_split_task_*.py`  
  Scripts used to reconstruct the exact train, validation, and test splits
  employed in the paper for each task, starting from the official UWF4DR dataset.

- `eval/`  
  Evaluation scripts that load pretrained models, apply the appropriate
  preprocessing, and compute performance metrics on the reconstructed test sets.

- `preprocessing/`  
  Deterministic preprocessing pipelines for spatial-domain and frequency-domain
  image representations.

- `models/`  
  Model architecture definitions and utility functions for loading pretrained
  models.  
  **No trained model weights are stored in this repository.**

---

## Interaction with Other Repository Components

The code in `src/` is intended to be used in conjunction with:

- `data/`  
  Provides split definition files (`data/splits/`) and documentation of the
  dataset structure.

- `requirements/`  
  Specifies the Python dependencies required for different model families.
  Separate environments are expected for different architectures.

- `docs/reproducibility.md`  
  Provides a detailed explanation of the experimental protocol and how the
  scripts in `src/` are used to reproduce it.

---

## Model Weights

Pretrained model weights corresponding to the experiments reported in the paper
are **not distributed** within this repository.

They are hosted separately on institutional servers (BiDA Lab) and are expected
to be downloaded and referenced locally by the user when running evaluation
scripts.

---

## Intended Usage

Typical usage of the code in `src/` follows this workflow:

1. Obtain authorized access to the official UWF4DR dataset.
2. Reconstruct the experimental splits using the `prepare_split_task_*.py` scripts.
3. Download pretrained model weights from the external hosting location.
4. Evaluate the models using the scripts provided in `eval/`.

Detailed, step-by-step instructions are provided in the repository documentation.
