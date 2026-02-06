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

## Dataset Source

The experiments reported in the paper are based on the **UWF4DR (Ultra-Widefield Imaging for Diabetic Retinopathy) Challenge dataset**, originally released in the context
of MICCAI 2024.

The dataset is distributed by the challenge organizers and must be obtained
separately. Access to the data requires registration to the challenge platform.
Official information and access instructions are available at:

https://codalab.lisn.upsaclay.fr/competitions/18605

---

## Purpose of This Repository

Due to licensing and data protection restrictions associated with the UWF4DR dataset,
the original ultra-widefield fundus images cannot be redistributed.

This repository therefore focuses on **reproducibility of the experimental protocol**
described in the paper. Specifically, it provides:

- fixed experimental split definitions used in the paper,
- scripts to reconstruct the exact train, validation, and test splits from the official dataset,
- preprocessing pipelines for spatial and frequency domains,
- evaluation scripts for deep learning models.

---

## Repository Structure

The repository is organized as follows:

- `data/`  
  Dataset documentation and fixed split definition files employed in the paper.  
  **No image data is included.**

- `src/`  
  Python source code for dataset split preparation, preprocessing, and model evaluation.
  Model architecture definitions are included, but no trained model weights are stored.
  Pretrained models used in the paper are hosted separately on institutional servers (BiDA Lab).

- `requirements/`  
  Dependency specifications for the different model families used in the study.

- `docs/`  
  Additional documentation related to reproducibility and the experimental protocol.

---

## Split Reconstruction and Usage

Once authorized access to the UWF4DR dataset is obtained, users can reproduce the exact
experimental splits used in the paper by running the provided
`prepare_split_task_x.py` scripts.

Detailed information about dataset structure and split reconstruction is provided in:

- `data/README.md`

---

## Reproducibility and Dependencies

The repository has been tested using **Python 3.10**.

Dependencies required for each family of models used in this study are specified in
the `requirements/` directory.

Different model families (e.g., CNN-based models, vision transformers, and retinal
foundation models) rely on distinct software stacks and are therefore intended to be
used in separate Python environments.

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
