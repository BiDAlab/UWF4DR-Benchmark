# Reproducibility Guide

This document describes how to reproduce the experimental protocol and results
reported in the paper:

**Exploring Deep Learning Methods and Ultra-Widefield Imaging for Diabetic Retinopathy**

The goal of this guide is to provide a clear and complete description of:
- the dataset usage and experimental splits,
- the preprocessing pipelines,
- the training and evaluation protocols,
- and the explainability analyses.

This repository follows standard reproducibility practices for medical imaging
and deep learning research. Due to dataset licensing restrictions, raw data
and trained model weights are not directly included in the repository.

---

## 1. Experimental Overview

The experiments are conducted on the UWF4DR dataset using ultra-widefield
fundus images. Three binary classification tasks are addressed:

- **Task 1 – Image Quality Assessment**: Gradable vs. Ungradable images  
- **Task 2 – Referable Diabetic Retinopathy (RDR)**: Non-referable vs. Referable DR  
- **Task 3 – Diabetic Macular Edema (DME)**: Absence vs. Presence of DME  

For each task, models are trained and evaluated using two image representations:
- Spatial domain (RGB images)
- Frequency domain (magnitude of the 2D Discrete Fourier Transform)

The experimental protocol includes multiple backbone architectures,
feature-level fusion strategies, and explainability analyses based on Grad-CAM.

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

Exact numerical reproducibility may vary slightly due to hardware and
software differences, but all methodological steps are fully specified.

---

## 4. Dependencies and Environments

Different model families were trained using separate Python environments
to ensure compatibility with their respective dependencies.

The exact requirements for each model family are provided in the
`requirements/` directory.

This design choice avoids version conflicts between architectures while
preserving full reproducibility of the experimental protocol.
