# Preprocessing Pipelines

## Overview

This directory contains the deterministic image preprocessing utilities used in the UWF4DR benchmark for evaluation and inference.

Two complementary input domains are supported:

- **Spatial domain (RGB)**: Standard ultra-widefield fundus images after cropping, resizing, and color normalization.
- **Frequency domain (DFT magnitude)**: Frequency-based representations derived from the magnitude of the
  two-dimensional Discrete Fourier Transform (2D DFT), with intensity values
  clipped at the 99th percentile.

These preprocessing blocks ensure consistency with the experimental protocol described in the paper and enable reproducible evaluation.

This module does **not** handle dataset splitting, model training, or augmentation strategies. It only defines reusable image transformations.

---

## Design Principles

The preprocessing module follows these principles:

- **Deterministic transformations**  
  All preprocessing operations are fixed and non-stochastic to ensure exact
  reproducibility of the evaluation pipeline.

- **Domain separation**  
  Spatial and frequency-domain pipelines are implemented independently.

- **Modular design**  
  Preprocessing is implemented as reusable building blocks that can be composed within model-specific evaluation pipelines.

- **Evaluation-oriented**  
  These transformations are intended for reproducible evaluation and inference. Training-time augmentations are intentionally excluded.

---

## File Structure

### `spatial.py`

Implements spatial-domain preprocessing for RGB images:

- Center crop to **800 × 800**
- Resize to the required input resolution
- Local mean subtraction for color normalization

---

### `frequency.py`

Implements frequency-domain preprocessing:

- Center crop to **800 × 800**
- Resize to the required input resolution
- 2D DFT magnitude computation
- Clipping at the **99th percentile**
- Min-max normalization

This representation is particularly useful for capturing blur and texture-related characteristics.

---

### `preprocess_factory.py`

Provides utility functions that combine:

- Domain preprocessing (spatial or frequency)
- Backbone-specific normalization (e.g., ImageNet preprocessing for CNN models)

This enables consistent input formatting across different model families while maintaining a unified preprocessing structure.

---

## Integration

The preprocessing functions defined in this directory are invoked internally by the evaluation pipelines in `src/eval/`.

The final input resolution and model-specific formatting are determined by the evaluation scripts, which compose these preprocessing building blocks as required by each architecture.

Users are not expected to call these functions directly unless extending or modifying the evaluation pipeline.

For a complete description of how preprocessing integrates into the experimental workflow, refer to:

- `src/eval/`
- `docs/reproducibility.md`

---

