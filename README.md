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
  - `splits/`: CSV files defining the exact train/validation/test partitions used in the experiments reported in the paper. No image data is included.
  - `fusion/`: Precomputed feature embeddings (per task, domain, and backbone) used for feature-level fusion experiments.

- `src/`
  - `eval/`: Evaluation scripts for CNNs (MobileNetV2, ResNet18), ViT-B/16, RETFound, and fusion (MLP) models.
  - `models/`: Model builders and loaders. Model architecture definitions are included, but no trained model weights are stored. Pretrained models used in the paper are hosted separately on institutional servers (BiDA Lab).
  - `preprocessing/`: Spatial and frequency-domain preprocessing pipelines.
  - `prepare_splits/`: Scripts to reconstruct the exact dataset partitions described in the paper.

- `requirements/`: Dependency specifications for the different model families used in the study.

- `docs/`: Additional documentation related to reproducibility and the experimental protocol.

---

## Quickstart

### 1️⃣ Reconstruct dataset splits

Once authorized access to the UWF4DR dataset is obtained, users can reconstruct the exact train/validation/test partitions used in the experiments by running the task-specific split preparation scripts located in `src/prepare_splits`:

```bash
python -m src.prepare_splits.prepare_split_task1 --output_dir <OUTPUT_DIR>
python -m src.prepare_splits.prepare_split_task2 --output_dir <OUTPUT_DIR>
python -m src.prepare_splits.prepare_split_task3 --output_dir <OUTPUT_DIR>
```

---

## Reproducibility and Dependencies

The repository has been tested using **Python 3.10**.

Dependencies required for each family of models used in this study are specified in
the `requirements/` directory.

Different model families (e.g., CNN-based models, vision transformers, and retinal
foundation models) rely on distinct software stacks and are therefore intended to be
used in separate Python environments.

### Python Version Notes

While the main repository (CNNs and ViT-B/16 models) has been tested with **Python 3.10**, the RETFound implementation is based on **Python 3.9** and TensorFlow 2.8.x, following the official RETFound repository.

For full compatibility and reproducibility of RETFound experiments, we recommend creating a dedicated **Python 3.9** environment when working with the dependencies listed in `requirements/retfound.txt`.

---

## Foundation Model: RETFound

This repository includes support for models fine-tuned from **RETFound**, the large-scale retinal foundation model introduced by:

Zhou et al., *"A foundation model for generalizable disease detection from retinal images"*, Nature, 2023.

Official Keras implementation:
https://github.com/wangseann/RETFound_MAE_keras

Within the framework of this work, RETFound is incorporated as one of the evaluated architectures and is used both as an individual classifier and as a feature extractor for the feature-level fusion experiments described in the paper.

### License

RETFound is released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Users of this repository must comply with the original RETFound license terms, particularly with respect to the non-commercial restriction.

This repository:

- does not redistribute the original pretrained RETFound weights,
- does not include code copied from the official RETFound repository,
- only provides scripts to load and evaluate fine-tuned models trained using the official RETFound codebase.

For full license details, please refer to the official RETFound repository.

### Requirements

The file `requirements/retfound.txt` is derived from the official RETFound `requirements.txt` file and preserves the original dependency versions necessary to instantiate and use the RETFound architecture.

### Implementation Details

In this work, RETFound was fine-tuned using the official `main_finetune.py` script (with minor modifications) provided in the original repository.

For evaluation and feature extraction within this repository, we:

- instantiate the ViT-Large architecture (`vit_large_patch16_224`) using `tfimm`,
- load the fine-tuned weights,
- extract the CLS token representation after the final normalization layer.

Image preprocessing follows the protocol described in the paper:

- center crop to 800×800,
- resize to 224×224,
- normalization to the [0, 1] range.

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
