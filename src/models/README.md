# Model Loading Utilities and Architecture Definitions

This directory contains model construction and loading utilities used by the evaluation pipelines of the UWF4DR benchmark.

It provides lightweight builders and standardized interfaces for loading pretrained models corresponding to the architectures evaluated in the associated paper.

---

## Scope

The purpose of this directory is to provide:

- Reproducible model construction utilities for the architectures used in the study.
- Standardized interfaces for loading pretrained models or weights.
- A clear separation between model definition and external model distribution.

This repository focuses on **evaluation and inference**.  
Training scripts, intermediate checkpoints, and experimental logs are out of scope.

---

## Supported Architectures

The following model families are implemented and supported:

- **Convolutional Neural Networks (CNNs)**
  - MobileNetV2
  - ResNet18

- **Vision Transformers**
  - ViT-B/16

- **Retinal Foundation Models**
  - RETFound

- **Feature-Level Fusion**
  - Multilayer Perceptron (MLP) used for fusion of extracted feature embeddings

Each architecture has corresponding model-building and/or loading utilities in this directory.

---

## Model Loading Behavior

Loaders support externally provided pretrained models in standard formats, including:

- Full Keras models (`.keras`)
- Weight files (e.g., `.h5`), depending on the architecture

Some architectures require specific third-party libraries (e.g., `classification_models`, `vit_keras`, `tfimm`).  
Dependencies are organized per model family under the `requirements/` directory.

Most architecture-specific configuration (e.g., input resolution, preprocessing composition, and evaluation logic) is handled by the evaluation scripts in `src/eval/`, while certain model-specific helpers are defined here.

---

## Model Weights Distribution

Due to file size constraints and licensing considerations, pretrained model
weights are **not distributed through this repository**.

Instead, the pretrained models corresponding to the experiments reported in the
paper are hosted externally on the official BiDA Lab servers.

Users are expected to:

1. Download the pretrained model files from the external hosting location.
2. Provide the local path to the model weights when running evaluation scripts.

---

## Pretrained Model Access

Pretrained model weights are hosted externally by BiDA Lab and are not included in this repository.

Download link:

https://bidalab.eps.uam.es/static/UWF4DR-Benchmark/Models.zip

### Download instructions

Using wget:

```bash
wget https://bidalab.eps.uam.es/static/UWF4DR-Benchmark/Models.zip
unzip Models.zip -d models/pretrained
```

Or using curl:

```bash
curl -O https://bidalab.eps.uam.es/static/UWF4DR-Benchmark/Models.zip
unzip Models.zip -d models/pretrained
```

You may also download the ZIP file directly using your browser and extract it locally.

---

## Design Notes

- Only final models corresponding to the results reported in the paper are intended to be shared.
- Intermediate checkpoints, training logs, and auxiliary artifacts are not distributed.
- Model utilities are designed to be compatible with externally hosted files without requiring modification of the repository structure.

---

## Relation to Evaluation Code

The model-loading utilities defined here are invoked internally by the evaluation scripts in `src/eval/`.

This directory provides modular model construction and loading components, while the evaluation scripts orchestrate preprocessing, model instantiation, and metric computation.

---
