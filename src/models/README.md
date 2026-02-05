# Pretrained Models

This directory documents the pretrained deep learning models associated with
the experiments reported in the paper.

Due to file size constraints, trained model weights are **not stored directly**
in this repository.

---

## Model Distribution

All pretrained models are made available through **GitHub Releases**.

Each release corresponds to a stable experimental configuration and includes:
- model weights,
- configuration files,
- evaluation metrics.

This strategy ensures efficient distribution while keeping the repository
lightweight and version-controlled.

---

## Naming Convention

Releases follow a descriptive naming convention indicating:
- the task (Quality, RDR, or DME),
- the input domain (RGB or Frequency),
- the model architecture.

Example:
v1.0-task2-rgb-retfound


---

## Notes

- Only final models corresponding to reported results are released.
- Intermediate checkpoints are not shared.
