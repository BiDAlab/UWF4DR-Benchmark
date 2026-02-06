# Preprocessing Pipelines

This directory contains the deterministic preprocessing pipelines applied to
input images before model evaluation.

The preprocessing steps implemented here are an integral part of the
experimental protocol described in the associated paper and are applied
consistently across all tasks and model architectures.

---

## Overview

Two complementary image representations are considered in the experiments:

- **Spatial domain**  
  Standard RGB ultra-widefield fundus images after resizing and
  color normalization.

- **Frequency domain**  
  Frequency-based representations derived from the magnitude of the
  two-dimensional Discrete Fourier Transform (2D DFT), with intensity values
  clipped at the 99th percentile.

The preprocessing pipelines implemented in this directory transform raw input
images into these representations in a fully deterministic manner.

---

## Design Principles

The preprocessing code follows these principles:

- **Deterministic behavior**  
  All preprocessing operations are fixed and non-stochastic to ensure exact
  reproducibility of the evaluation pipeline.

- **Shared preprocessing logic**  
  The same core preprocessing operations are applied across tasks and model
  families. Domain-specific transformations (spatial or frequency) are
  implemented within the preprocessing modules, while task- and model-specific
  requirements (such as input image resolution) are defined and handled by the
  evaluation scripts.

- **Explicit domain separation**  
  Spatial-domain and frequency-domain preprocessing pipelines are implemented
  as separate modules to avoid ambiguity and ensure clarity.

---

## Implementation Notes

- Preprocessing functions operate on individual images and return tensors ready
  for model inference.
- Input image resizing is performed according to task-specific requirements
  defined in the evaluation scripts.
- No data augmentation is applied at evaluation time.
- No dataset-specific assumptions beyond image format and size are hard-coded.

---

## Usage

The preprocessing functions in this directory are invoked internally by the
evaluation scripts in `src/eval/`.

Users are not expected to call these functions directly unless extending or
modifying the evaluation pipeline.

For details on how preprocessing integrates into the full experimental workflow,
refer to `docs/reproducibility.md`.
