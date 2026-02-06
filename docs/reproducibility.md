# Reproducibility Guide

This document describes how to reproduce the experimental protocol and results reported in the paper:

**Exploring Deep Learning and Ultra-Widefield Imaging for Diabetic Retinopathy and Macular Edema**

The goal of this guide is to provide a clear and complete description of:

- the dataset usage and experimental splits,
- the preprocessing pipelines,
- and the evaluation protocols.

This guide focuses on reproducibility of the experimental design and evaluation
pipeline. While exact numerical results may vary slightly due to hardware and
software differences, all methodological steps and experimental decisions are
fully specified.

---

## 1. Experimental Overview

The experiments are conducted using the **UWF4DR dataset**, composed of ultra-widefield (UWF) fundus images. Three binary classification tasks are addressed:

- **Task 1 – Image Quality Assessment**: Gradable vs. Ungradable images  
- **Task 2 – Referable Diabetic Retinopathy (RDR) Identification**: Non-referable vs. Referable DR  
- **Task 3 – Diabetic Macular Edema (DME) Identification**: Absence vs. Presence of DME  

For each task, models are trained and evaluated using two image representations:

- Spatial domain (RGB images)  
- Frequency domain (magnitude of the 2D Discrete Fourier Transform, with values clipped at the 99th percentile)

The experimental protocol described in the paper considers multiple backbone architectures, including convolutional neural networks (MobileNetV2 and ResNet18), vision transformer models (ViT-B/16), and retinal foundation models (RETFound, originally described in https://www.nature.com/articles/s41586-023-06555-x, with an official open-source Keras implementation available at https://github.com/uw-biomedical-ml/RETFound_MAE).

---

## 2. Repository Structure

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

## 3. Dataset Origin and Access

The data used in this work originate from the **UWF4DR Challenge**, an official international challenge organized in the context of the **MICCAI 2024 conference**.

The UWF4DR Challenge is hosted on the **CodaLab** platform, which manages dataset access, evaluation protocols, and leaderboard submissions. The official competition page is available at:

👉 **https://codalab.lisn.upsaclay.fr/competitions/18605**

Access to the dataset is subject to the terms and conditions defined by the challenge organizers. Users must register on the CodaLab platform and obtain the data directly through the official competition interface.

This repository is **not affiliated** with the organization of the challenge and does not host or redistribute any part of the dataset.

---

## 3.1 Handling of Annotations and Licensing Constraints

Due to dataset licensing constraints imposed by the UWF4DR Challenge, official annotations and ground-truth labels are **not redistributed** in this repository.

Instead, this repository provides:

- **split definition files** (located in `data/splits/`), and  
- **dataset preparation scripts** (located in `src`,

which allow users to reconstruct **exactly the same train, validation, and test splits** used in the paper, starting from the officially downloaded datasets.

The split CSV files included in the repository:

- do **not** contain any medical labels,  
- only specify:
  - the image identifier, and  
  - the split assignment (train, validation, or test).

Final labels are obtained locally by the user by combining:

- the official UWF4DR ground-truth CSV files provided with the dataset, and  
- the split definitions included in this repository.

This design ensures:

- compliance with the UWF4DR Challenge licensing terms,  
- no redistribution of sensitive medical annotations,  
- and full reproducibility of the experimental protocol.

---

## 4. Dataset Split Reconstruction

One script is provided per task to reconstruct the exact dataset partitions used in the paper:

- **Task 1 – Image Quality Assessment**: `prepare_split_task_1.py`  
- **Task 2 – Referable DR Identification**: `prepare_split_task_2.py`  
- **Task 3 – DME Identification**: `prepare_split_task_3.py`  

Each script:

- reads the official UWF4DR ground-truth annotations locally,  
- applies the fixed split definition used in the paper,  
- resolves the correct image paths from the downloaded datasets,  
- and optionally saves the resulting splits as CSV files.

**Task-specific considerations:**

- For **Task 1**, some images physically reside in the Task 2/3 folders; this is handled automatically by the script.  
- For **Task 2**, all images in the Task 2/3 dataset are included.  
- For **Task 3**, only images with an available DME annotation (non-empty DME label in the official CSV) are included.

When executed correctly, the scripts generate `train.csv`, `validation.csv`, and `test.csv` files whose sizes exactly match those reported in the paper.

---

## 5. Dependencies and Environments

Different model families were trained using separate Python environments to ensure compatibility with their respective dependencies.

The exact requirements for each model family are provided in the `requirements/` directory.

Some Python dependencies are installed directly from GitHub repositories. Therefore, **Git must be available** in the system before installing the requirements.

If using conda, Git can be installed with:

```bash
conda install -c conda-forge git
