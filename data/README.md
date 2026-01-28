# Dataset Information

This repository uses the UWF4DR dataset, released as part of the MICCAI 2024
Ultra-Widefield Imaging for Diabetic Retinopathy (UWF4DR) Challenge.

Due to data protection, licensing, and privacy regulations, the original
ultra-widefield fundus images cannot be publicly distributed within this
repository.

---

## Dataset Access

The UWF4DR dataset is publicly described and documented by the challenge
organizers. Interested researchers should refer to the official sources
for dataset access and licensing terms: https://codalab.lisn.upsaclay.fr/competitions/18605

Once access is granted, users can reproduce the experiments reported in the
paper by organizing the data according to the structure described below.

---

## Expected Directory Structure

After obtaining authorized access to the dataset, images should be organized
as follows:

```
data/
├── train/
├── val/
└── test/
```

Each directory should contain the corresponding ultra-widefield fundus images
for the associated experimental split.

---

## Experimental Splits

The train, validation, and test splits used in the paper follow an independent
experimental protocol defined by the authors, as the official challenge test
set is not publicly accessible and cannot be used for independent evaluation.

The exact split definitions are provided as CSV files, specifying the image
identifiers assigned to each subset.

---

## Notes

- No image data is included in this repository.
- The provided splits ensure reproducibility of all reported experiments.
- Any use of the dataset must comply with the original licensing terms.
