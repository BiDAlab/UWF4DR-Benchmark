# Model Architectures and Loading Utilities (`models/`)

This directory contains model architecture definitions and utility functions
used to load pretrained deep learning models for evaluation.

The code in this directory supports the backbone architectures considered in
the experimental protocol described in the associated paper, but **does not
store any trained model weights**.

---

## Scope

The purpose of this directory is to provide:

- reproducible implementations of the model architectures used in the study,
- standardized interfaces for loading pretrained weights,
- and a clear separation between model definition and model distribution.

This directory is **not** intended to host trained models, checkpoints, or
experimental results.

---

## Supported Architectures

The following model families are supported or planned, in accordance with the
paper:

- **Convolutional Neural Networks (CNNs)**  
  - MobileNetV2  
  - ResNet18  

- **Vision Transformer models**  
  - ViT-B/16  

- **Retinal foundation models**  
  - RETFound  

Architecture-specific details (e.g., input resolution, preprocessing
requirements, and weight compatibility) are handled by the corresponding
evaluation scripts.

---

## Model Weights Distribution

Due to file size constraints and licensing considerations, pretrained model
weights are **not distributed through this repository**.

Instead, the pretrained models corresponding to the experiments reported in the
paper are hosted separately on **institutional servers (BiDA Lab)**.

Users are expected to:

1. Download the pretrained model files from the external hosting location.
2. Provide the local path to the model weights when running evaluation scripts.

Details regarding model availability and access instructions will be provided
separately by the authors.

---

## Design Notes

- Only final models corresponding to the results reported in the paper are
  intended to be shared.
- Intermediate checkpoints, training logs, and auxiliary artifacts are not
  distributed.
- The code in this directory is written to be compatible with externally hosted
  model files without requiring modification of the repository structure.

---

## Relation to Evaluation Code

The model loading utilities defined here are used by the evaluation scripts in
`src/eval/`.

As additional evaluation pipelines (e.g., for ViT or RETFound) are incorporated
into the repository, corresponding model-loading logic will be added to this
directory.
