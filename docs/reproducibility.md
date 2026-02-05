# Reproducibility Guide

This document describes how to reproduce the experimental protocol and results reported in the paper:

**Exploring Deep Learning Methods and Ultra-Widefield Imaging for Diabetic Retinopathy**

The goal of this guide is to provide a clear and complete description of:

- the dataset usage and experimental splits,
- the preprocessing pipelines,
- the training and evaluation protocols,
- and the explainability analyses.

This repository follows standard reproducibility practices for medical imaging and deep learning research. Due to dataset licensing restrictions, raw data and trained model weights are not directly included in the repository.

---

## 1. Experimental Overview

The experiments are conducted on the **UWF4DR dataset** using ultra-widefield fundus images. Three binary classification tasks are addressed:

- **Task 1 – Image Quality Assessment**: Gradable vs. Ungradable images  
- **Task 2 – Referable Diabetic Retinopathy (RDR)**: Non-referable vs. Referable DR  
- **Task 3 – Diabetic Macular Edema (DME)**: Absence vs. Presence of DME  

For each task, models are trained and evaluated using two image representations:

- Spatial domain (RGB images)
- Frequency domain (magnitude of the 2D Discrete Fourier Transform)

The experimental protocol includes multiple backbone architectures, feature-level fusion strategies, and explainability analyses based on Grad-CAM.

---

## 2. Repository Structure

The repository is organized as follows:

- `configs/`: configuration files for experiments and models  
- `data/`: placeholder directory for dataset organization (data not included)  
- `docs/`: documentation files, including this reproducibility guide  
- `experiments/`: experiment definitions and execution scripts  
- `models/`: model definitions and architecture wrappers  
- `results/`: output metrics, logs, and visualizations (excluding large files)  
- `src/`: source code for data processing, training, evaluation, and explainability  

---

## 3. Reproducibility Scope

This guide allows reproduction of:

- the experimental data splits,
- the preprocessing pipelines,
- the training and evaluation procedures,
- and the reported quantitative and qualitative results.

Exact numerical reproducibility may vary slightly due to hardware and software differences, but all methodological steps and experimental decisions are fully specified.

---

## 4. Dataset Access and Label Handling

Due to dataset licensing constraints, **official UWF4DR annotations are not redistributed in this repository**.

Instead, this repository provides **split definition files** (located in `data/splits/`) and **dataset preparation scripts** that allow users to reconstruct *exactly the same train, validation, and test splits* used in the paper, starting from the officially released datasets.

The split CSV files included in the repository:

- **do not contain any medical labels**,  
- only specify:
  - the image identifier, and
  - the split assignment (`train`, `validation`, or `test`).

Final labels are obtained locally by the user by combining:
1. the official UWF4DR ground-truth CSV files, and  
2. the split definitions provided in this repository.

This design ensures:
- compliance with dataset licensing terms,
- no redistribution of sensitive medical annotations,
- and full reproducibility of the experimental protocol.

---

## 5. Dataset Split Reconstruction

One script is provided per task to reconstruct the exact dataset partitions used in the paper:

- **Task 1 – Image Quality Assessment**: `prepare_split_task_1.py`
- **Task 2 – Referable DR Identification**: `prepare_split_task_2.py`
- **Task 3 – DME Identification**: `prepare_split_task_3.py`

Each script:
- reads the official UWF4DR ground-truth annotations locally,
- applies the fixed split definition used in the paper,
- resolves the correct image paths from the downloaded datasets,
- and optionally saves the resulting splits as CSV files.

Task-specific considerations:

- For **Task 1**, some images physically reside in the Task 2/3 folders; this is handled automatically by the script.
- For **Task 2**, all images in the Task 2/3 dataset are included.
- For **Task 3**, only images with an available DME annotation (non-empty DME label in the official CSV) are included.

When executed correctly, the scripts generate `train.csv`, `validation.csv`, and `test.csv` files whose sizes exactly match those reported in the paper.

---

## 6. Dependencies and Environments

Different model families were trained using separate Python environments to ensure compatibility with their respective dependencies.

The exact requirements for each model family are provided in the `requirements/` directory.

This design choice avoids version conflicts between architectures while preserving full reproducibility of the experimental protocol.

This design choice avoids version conflicts between architectures while
preserving full reproducibility of the experimental protocol.
