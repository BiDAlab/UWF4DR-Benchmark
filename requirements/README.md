# Requirements

This directory contains Python dependency files for the different model families used in the experiments.

Each file is intended to be installed in a **separate Python environment**, as different model families rely on distinct and potentially incompatible software stacks.

Unless otherwise specified, these environments were tested with **Python 3.10**.

Users should select and install only the requirements file corresponding to the model family they intend to evaluate.

## cnn.txt
Dependencies for convolutional neural network (CNN) models. The CNN architectures used in this study include MobileNetV2 and ResNet18.

This environment can also be used to run the MLP-based feature-level fusion experiments, as fusion scripts rely on TensorFlow, NumPy, pandas, and scikit-learn, which are already included in the CNN requirements.

## vit.txt
Dependencies for vision transformer (ViT) models.

## retfound.txt
Dependencies for RETFound-based retinal foundation models. This environment follows the original RETFound dependency versions and is intended to be used with Python 3.9.
