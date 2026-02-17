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

This repository is organized as follows:

- `data/`
  - `fusion/`: Precomputed feature embeddings (per task, domain, and backbone) used for feature-level fusion experiments.
  - `splits/`: CSV files defining the exact train/validation/test partitions used in the experiments reported in the paper. No image data is included.

- `docs/`: Additional documentation related to reproducibility and the experimental protocol.

- `requirements/`: Dependency specifications for the different model families used in the study.

- `src/`
  - `eval/`: Evaluation scripts for CNNs (MobileNetV2, ResNet18), ViT-B/16, RETFound, and feature-level fusion (MLP) models.
  - `models/`: Model builders and loaders. Model architecture definitions are included, but no trained model weights are stored. Pretrained models used in the paper are hosted separately on institutional servers (BiDA Lab).
  - `prepare_splits/`: Scripts to reconstruct the exact dataset partitions described in the paper.
  - `preprocessing/`: Spatial and frequency-domain preprocessing pipelines.

---

## Quickstart

### 1️⃣ Reconstruct dataset splits

Once authorized access to the UWF4DR dataset is obtained, users can reconstruct the exact train/validation/test partitions used in the experiments by running the task-specific split preparation scripts located in `src/prepare_splits`. This step only requires a minimal Python environment with `pandas` installed and can be executed independently of the deep learning environments used for model evaluation.

Each task may require additional dataset-specific arguments. For full documentation and argument descriptions, see src/prepare_splits/README.md. You can also inspect the available options using: 
```bash
python -m src.prepare_splits.prepare_split_task_<task_number> -h
```

### 2️⃣ Install dependencies

Different model families (CNNs, ViT-B/16, and RETFound) have separate dependency specifications located in the `requirements/` directory.

Create and activate the appropriate environment before running any evaluation script.

For detailed installation instructions and environment recommendations, see `requirements/README.md`.


### 3️⃣ Evaluate trained models

Once dataset partitions are prepared and model weights are available, trained models can be evaluated using the task-specific scripts located in `src/eval`.

Each evaluation script corresponds to a specific model family (CNNs, ViT-B/16, RETFound, and MLP-based feature-level fusion models) and may require additional arguments such as model paths, domain selection (spatial or frequency), and split directory locations.

For full usage details and argument descriptions, see `src/eval/README.md`

You can also inspect available options using:
```bash
python -m src.eval.eval_<model_family> -h
```

---

## Reproducibility and Dependencies

The repository has been tested with **Python 3.10**. Dependencies for each model family are specified in the `requirements/` directory and are intended to be installed in separate Python environments.

CNNs, ViT-B/16 and MLP-based feature-level fusion components were tested with Python 3.10. RETFound follows the official implementation based on Python 3.9 and TensorFlow 2.8.x; for full compatibility, we recommend creating a dedicated Python 3.9 environment when using `requirements/retfound.txt`.

---

## Foundation Model: RETFound

This repository includes support for models fine-tuned from **RETFound**, the large-scale retinal foundation model introduced by Zhou et al. in *"A foundation model for generalizable disease detection from retinal images"* (Nature, 2023).

The official Keras implementation of RETFound is available in the following repository:
https://github.com/wangseann/RETFound_MAE_keras

Within the framework of this work, RETFound is incorporated as one of the evaluated architectures and is used both as an individual classifier and as a feature extractor for MLP-based feature-level fusion experiments described in the associated paper.

### License

RETFound is released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Users of this repository must comply with the original RETFound license terms, particularly with respect to the non-commercial restriction.

This repository:

- does not redistribute the original pretrained RETFound weights,
- does not redistribute source code from the official RETFound repository,
- only provides wrappers to load and evaluate fine-tuned models trained using the official RETFound codebase.

For full license details, please refer to the official RETFound repository.

### Requirements

The file `requirements/retfound.txt` is derived from the official RETFound `requirements.txt` file and preserves the original dependency versions necessary to instantiate and use the RETFound architecture.

### Implementation Notes

The fine-tuning procedure described in the associated paper was conducted using the official `main_finetune.py` script (with minor modifications) from the original RETFound repository.

Technical details regarding model instantiation, weight loading, and feature extraction within this repository are documented in `src/models/README.md`.

---

## Scope and Limitations

This repository does not include:

- The UWF4DR image data.
- Trained model weights.
- Precomputed predictions or reported performance metrics.

It includes dataset split definitions and precomputed feature embeddings for feature-level fusion experiments, but no raw images or trained models.

---

## License and Citation

The repository license will be specified in a future update.

The formal citation for the paper will be provided upon acceptance.
