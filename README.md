# UWF4DR-Benchmark

This repository provides the experimental protocol, model configurations, and
reproducibility resources associated with the paper:

**Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**

The study investigates state-of-the-art deep learning approaches for the analysis
of ultra-widefield (UWF) fundus images across three clinically relevant tasks:

- image quality assessment,
- referable diabetic retinopathy (RDR) identification,
- diabetic macular edema (DME) identification.

---

## Repository Contents

The repository is organized as follows:

- `configs/`: configuration files for model training and evaluation  
- `data/`: dataset documentation and expected directory structure (no images included)  
- `docs/`: detailed reproducibility documentation  
- `experiments/`: experiment definitions and execution scripts  
- `models/`: model architectures and wrappers  
- `results/`: evaluation outputs and visualizations (excluding large files)  
- `src/`: source code for preprocessing, training, evaluation, and explainability  

---

## Dataset Availability

Due to data protection and licensing restrictions, the original UWF fundus images
and official annotations are **not included** in this repository.

Instead, this repository provides fixed split definition files and Python scripts
that allow users to reconstruct **exactly the same train, validation, and test
splits** used in the paper, starting from the official UWF4DR dataset releases.

Information about dataset usage and directory structure is provided in:

- `data/README.md`

---

## Reproducibility

Detailed instructions to reproduce the experimental protocol and results are
available in:

- `docs/reproducibility.md`

This includes:
- dataset preparation and split reconstruction,
- preprocessing pipelines,
- training and evaluation procedures,
- and explainability analyses.

Python dependencies for the CNN-based models are listed in:

- `requirements/cnn.txt`

---

## License and Citation

License and citation information will be added upon paper acceptance.
