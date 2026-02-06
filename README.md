# UWF4DR-Benchmark

This repository provides the experimental protocol, data split definitions, and
evaluation code associated with our submission to the **UWF4DR Challenge**
(Ultra-Widefield Imaging for Diabetic Retinopathy), organized in the context of
the following paper:

**Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**

The UWF4DR Challenge was originally hosted as part of MICCAI 2024 and focuses on
the analysis of ultra-widefield (UWF) fundus images for multiple clinically
relevant tasks.

In this work, we study deep learning approaches for three tasks:

- **Task 1**: Image quality assessment  
- **Task 2**: Referable diabetic retinopathy (RDR) identification  
- **Task 3**: Diabetic macular edema (DME) identification  

---

## Purpose of This Repository

Due to licensing and data protection restrictions, the official UWF4DR dataset
cannot be redistributed. This repository therefore focuses on **reproducibility**
rather than data hosting.

Specifically, it provides:

- fixed experimental split definitions used in the paper,
- scripts to reconstruct the exact train/validation/test splits from the official dataset,
- preprocessing pipelines for spatial and frequency domains,
- evaluation scripts for convolutional neural network (CNN) models.

All code has been validated from a clean environment using dummy models to ensure
that the full pipeline can be executed by external users.

---

## Repository Structure

The repository is organized as follows:

- `data/`  
  Documentation of the dataset usage and fixed split definitions employed in the paper.
  No image data is included.

- `src/`  
  Python source code for dataset split preparation, preprocessing, and model evaluation.
  This directory contains architecture definitions and evaluation pipelines, but no
  trained model weights.

- `requirements/`  
  Dependency specifications for the different model families used in the study.

- `docs/`  
  Additional documentation related to reproducibility and experimental protocol.

---

## Dataset Availability and Split Reconstruction

The official UWF4DR dataset must be obtained directly from the competition organizers.
Once access is granted, users can reproduce the exact experimental splits used in the
paper by running the provided `prepare_split_task_x.py` scripts.

Detailed information about dataset structure and split reconstruction is provided in:

- `data/README.md`

---

## Reproducibility and Dependencies

The repository has been tested using **Python 3.10**.

Dependencies required to evaluate CNN-based models (e.g., MobileNetV2 and ResNet18)
are listed in:

- `requirements/cnn.txt`

The evaluation pipeline for CNNs is fully functional and validated.

Support for transformer-based architectures, including **ViT-B/16** and
self-supervised retinal foundation models such as **RETFound**, is planned and
will be added in future updates.

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
