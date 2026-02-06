# UWF4DR-Benchmark

This repository provides the experimental protocol, data split definitions, and
evaluation code corresponding to the study presented in the paper:

**Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**  

The study investigates deep learning approaches for the analysis of ultra-widefield
(UWF) fundus images across three clinically relevant tasks:

- **Task 1**: Image quality assessment  
- **Task 2**: Referable diabetic retinopathy (RDR) identification  
- **Task 3**: Diabetic macular edema (DME) identification  

---

## Purpose of This Repository

Due to licensing and data protection restrictions, the ultra-widefield fundus images
used in the study cannot be redistributed. This repository therefore focuses on
**reproducibility of the experimental protocol**, rather than data hosting.

Specifically, it provides:

- fixed experimental split definitions used in the paper,
- scripts to reconstruct the exact train, validation, and test splits,
- preprocessing pipelines for spatial and frequency domains,
- evaluation scripts for convolutional neural network (CNN) models.

---

## Repository Structure

The repository is organized as follows:

- `data/`  
  Dataset documentation and fixed split definition files employed in the paper.  
  **No image data is included.**

- `src/`  
  Python source code for dataset split preparation, preprocessing, and model evaluation.  
  Model architecture definitions are included, but **no trained model weights** are stored.

- `requirements/`  
  Dependency specifications for the different model families used in the study.

- `docs/`  
  Additional documentation related to reproducibility and the experimental protocol.

---

## Dataset Source and Split Reconstruction

The experiments in this repository are based on the **UWF4DR Challenge dataset**
(Ultra-Widefield Imaging for Diabetic Retinopathy), originally released in the context
of MICCAI 2024.

The dataset is distributed by the challenge organizers and must be obtained
separately. Official information and access instructions are available at:

https://codalab.lisn.upsaclay.fr/competitions/18605

Once authorized access to the dataset is obtained, users can reproduce the exact
experimental splits used in the paper by running the provided
`prepare_split_task_x.py` scripts.

Detailed information about dataset structure and split reconstruction is provided in:

- `data/README.md`

---

## Reproducibility and Dependencies

The repository has been tested using **Python 3.10**.

Dependencies required to evaluate CNN-based models (e.g., MobileNetV2 and ResNet18)
are listed in:

- `requirements/cnn.txt`

The CNN evaluation pipeline is fully functional and has been validated across all
tasks and input domains.

Support for transformer-based architectures, including **ViT-B/16**, as well as
retinal foundation models such as **RETFound**, is planned and will be incorporated
in future updates.

---

## Scope and Limitations

This repository does **not** include:

- the UWF4DR image data,
- trained model weights,
- precomputed experimental results.

Trained models used in the paper will be hosted separately on institutional servers
(BiDA Lab) and made available for evaluation when possible.

---

## License and Citation

License and citation information will be added upon paper acceptance.

