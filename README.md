# Huggingface-deployement


# Hybrid U-Net and Transformer Architecture for Multi-Class Classification of Cervical Cells in LBC Images

## Model Overview

This repository contains the deployment version of a validated computer-vision model developed for multi-class classification of cervical cells in Liquid-Based Cytology (LBC) images.

The model classifies cervical-cell images into four classes:

* **NILM** — Negative for Intraepithelial Lesion or Malignancy
* **LSIL** — Low-Grade Squamous Intraepithelial Lesion
* **HSIL** — High-Grade Squamous Intraepithelial Lesion
* **SCC** — Squamous Cell Carcinoma

The deployed model combines **cell segmentation, morphological information, and transformer-based visual features** to produce the final classification.

---

## Model Pipeline

The inference pipeline follows the same processing flow used during model validation:

```text
Input RGB Cervical-Cell Image
            ↓
       Preprocessing
            ↓
       U-Net Segmentation
            ↓
     Predicted Binary Mask
            ↓
    Morphology Extraction
            ↓
      Morphology Features
            ↓
       Vision Transformer
            ↓
   Visual Representation
            ↓
 Morphology-Guided Fusion /
      Cross-Attention
            ↓
       Classification
            ↓
 ┌────────┬────────┬────────┬────────┐
 │  NILM  │  LSIL  │  HSIL  │   SCC  │
 └────────┴────────┴────────┴────────┘
```

The deployment reproduces the validated inference pipeline rather than using a separate simplified model.

---

## Input

The model expects a **cervical-cell microscopy/LBC RGB image**.

The input image is processed automatically using the same preprocessing configuration used during model validation.

The deployed system is intended for compatible cytology microscopy images and is **not designed for arbitrary photographs, skin images, selfies, or photographs of the cervix**.

---

## Output

The model returns a classification result containing the predicted class and model scores.

Example:

```json
{
  "predicted_class": "HSIL",
  "class_scores": {
    "NILM": 0.01,
    "LSIL": 0.04,
    "HSIL": 0.91,
    "SCC": 0.04
  },
  "confidence": 0.91
}
```

The exact output fields depend on the final inference implementation.

The reported score represents a **model confidence/score** and should not automatically be interpreted as a clinically calibrated probability.

---

## Architecture

The model is based on a hybrid computer-vision architecture.

### 1. U-Net Segmentation

The input RGB image is processed by a U-Net-based segmentation component to identify the cellular region.

The segmentation output is a binary mask.

```text
RGB Image
   ↓
U-Net
   ↓
Binary Cell Mask
```

### 2. Morphological Feature Extraction

The predicted mask is used to calculate morphological characteristics of the cell.

The project includes two morphology representations:

* **M1** — 14 morphological features
* **M2** — 22 morphological features

The exact representation used by this deployed checkpoint is documented in the model configuration.

### 3. Vision Transformer

The RGB image is also processed by the Vision Transformer to obtain visual representations of the cell.

The transformer captures visual patterns that complement the explicit morphological measurements.

### 4. Feature Fusion

The morphology and visual information are combined using the fusion mechanism implemented in the validated model.

Where applicable, morphology information guides the visual representation through the implemented fusion/cross-attention mechanism.

### 5. Classification

The final representation is passed to the classification head, which predicts one of four classes:

```text
NILM
LSIL
HSIL
SCC
```

---

## Preprocessing

The deployment uses the same preprocessing configuration established during model validation.

This includes the required:

* image loading
* RGB conversion
* resizing
* normalization
* tensor preparation
* segmentation preprocessing
* morphology preprocessing

Training-time augmentation is **not randomly applied to user images during normal inference**.

The deployment package contains the required preprocessing configuration and code so that a new image can be processed consistently.

---

## Morphology Processing

Morphological features are calculated from the predicted segmentation mask.

The morphology pipeline must preserve:

* feature definitions
* feature ordering
* feature scaling
* trained scaler parameters
* morphology representation version

If a training-fitted scaler is required, the deployment uses the existing fitted scaler rather than fitting a new scaler to an individual input image.

---

## Model Validation

The deployed checkpoint was selected following a separate final validation process.

The final evaluation was performed using the project's held-out test set.

### Evaluation Metrics

The following metrics are reported for the validated model:

| Metric            |                       Result |
| ----------------- | ---------------------------: |
| Accuracy          | **[INSERT MODULE 1 RESULT]** |
| Macro-F1          | **[INSERT MODULE 1 RESULT]** |
| Balanced Accuracy | **[INSERT MODULE 1 RESULT]** |

### Per-Class Performance

| Class | Precision |   Recall | F1-score |
| ----- | --------: | -------: | -------: |
| NILM  |  [RESULT] | [RESULT] | [RESULT] |
| LSIL  |  [RESULT] | [RESULT] | [RESULT] |
| HSIL  |  [RESULT] | [RESULT] | [RESULT] |
| SCC   |  [RESULT] | [RESULT] | [RESULT] |

**Do not replace these placeholders with estimated values.** Insert the exact results produced by the final validation module.

---

## Dataset

The model was developed using cervical-cell LBC datasets containing four classes:

```text
NILM
LSIL
HSIL
SCC
```

The original research dataset and its preprocessing/augmentation resources are used during model development and validation.

The complete training dataset is **not required for normal inference**.

A new prediction requires only a compatible input image and the model's inference dependencies.

---

## Deployment

The purpose of this repository is to provide a reproducible inference version of the validated computer-vision model.

The deployment package contains the components required for:

```text
New Image
   ↓
Preprocessing
   ↓
Segmentation
   ↓
Morphology
   ↓
Visual Feature Extraction
   ↓
Feature Fusion
   ↓
Classification
```

The model does not require access to Google Colab or Google Drive for normal inference.

---

## Intended Use

This model is intended for:

* research
* academic experimentation
* computer-vision research
* cervical-cell image classification
* demonstration of hybrid deep-learning architectures
* decision-support research

It is particularly intended to demonstrate the integration of segmentation, morphology, and transformer-based visual features for cervical-cell classification.

---

## Limitations

This model has several important limitations.

### Cell-Level Classification

The model performs **cell-level image classification**.

A prediction of HSIL or SCC should not be interpreted as a diagnosis of a patient with cervical cancer.

### Dataset Distribution

Performance may change when the model receives images that differ substantially from the training data in:

* imaging equipment
* staining procedures
* image resolution
* acquisition conditions
* cell preparation
* population
* laboratory protocols

### Input Requirements

The model requires a compatible cervical-cell microscopy/LBC image.

It should not be applied to arbitrary clinical photographs.

### Model Uncertainty

A classification score does not guarantee correctness.

Incorrect predictions can occur, particularly for visually similar cellular classes.

### Clinical Use

This model has not been presented as a standalone diagnostic device.

Any real clinical application would require additional validation, clinical evaluation, regulatory assessment, and professional oversight.

---

## Safety Statement

**This model is a research decision-support system and is not a standalone medical diagnostic system.**

The computer-vision prediction should be treated as a computational finding rather than a definitive clinical diagnosis.

The model should not replace evaluation by a qualified cytologist, pathologist, or other appropriate healthcare professional.

---

## Model Version

```text
Model: Hybrid U-Net and Transformer Cervical Cell Classifier
Version: [INSERT VALIDATED MODEL VERSION]
Validation Version: [INSERT MODULE 1 VERSION]
```

The deployed model should correspond exactly to the checkpoint that passed the project's final validation gate.

Future model modifications must be separately validated before replacing this version.

---

## Citation

If this model is used in academic work, please cite the associated thesis:

**“Hybrid U-Net and Transformer Architecture for Multi-Class Classification of Cervical Cells in LBC Images.”**

---

## Disclaimer

This repository provides an academic research implementation of a computer-vision model for cervical-cell classification.

It is not intended to provide medical advice, diagnosis, treatment recommendations, or definitive clinical conclusions.
