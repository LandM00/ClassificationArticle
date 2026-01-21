# ClassificationArticle

This repository contains the data, code and results associated with the classification study on plant species and growth stages under different light treatments.

---

## Overview

This repository provides:
- a light version of the experimental image dataset,
- the Python code used for model training and evaluation,
- the performance metrics obtained from cross-validation experiments.

The materials are released to support transparency, reproducibility and methodological reuse.

---

## Contents

The repository includes the following main components:

1. `DatasetLight.zip` – light version of the image dataset  
2. `ClassificationAlgo_def.py` – Python training and evaluation script  
3. `Metrics_CNN_results.xlsx` – performance metrics from cross-validation  

---

## 1. DatasetLight.zip

### Description

`DatasetLight.zip` represents a *light version* of the full experimental dataset used in the associated study. It contains a representative subset of the original images and is provided to support transparency, reproducibility, and methodological understanding, while limiting data volume.

The light dataset includes RGB images of four plant species acquired under controlled conditions and different light treatments. The images were selected to preserve the diversity of species, growth stages, and illumination conditions present in the full dataset.

### Directory structure

The dataset is organized according to the following hierarchical structure:

Species
└── Growth stage
└── Light treatment
└── Image files (.jpg)


Where:

- **Species** ∈ {Aubergine, Basil, Cucumber, Tomato}  
- **Growth stage** ∈ {GerminationStage, VegetativeStage}  
- **Light treatment** ∈ {RB1, RB3, RB5, RB7, RB9}  

Each image belongs to exactly one species, one growth stage, and one light treatment.

### Data acquisition and preprocessing

Images were acquired in controlled experimental conditions using a fixed imaging setup.  
All images are provided in JPEG format.  
During model training, images were resized to 256 × 192 pixels and normalized to the [0, 1] range.

---

## 2. ClassificationAlgo_def.py

### Description

`ClassificationAlgo_def.py` is the Python script that implements the training and evaluation pipeline of a multi-task convolutional neural network (CNN) for the simultaneous classification of plant species and growth stages in the presence of different light treatments.

### Main functionalities

The code performs the following main steps:

- data loading and labeling,  
- image preprocessing and data augmentation,  
- definition of a multi-task CNN architecture,  
- five-fold cross-validation training,  
- performance metric computation,  
- confusion matrix generation,  
- export of results to Excel files and image plots.  

### Computational environment

The code is written in Python and relies on the following main libraries:

- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- Pandas  
- Matplotlib  

GPU acceleration and mixed-precision training are supported.

---

## 3. Metrics_CNN_results.xlsx

### Description

`Metrics_CNN_results.xlsx` contains the performance metrics obtained from the evaluation of the CNN model for plant species classification under different experimental treatments.

The dataset reports the results of a five-fold cross-validation procedure, together with the average performance across folds, and was used to support the quantitative analysis presented in the associated scientific article.

### File structure

The Excel file is organized into six worksheets:

- Fold_1  
- Fold_2  
- Fold_3  
- Fold_4  
- Fold_5  
- Media_fold  

Each worksheet corresponds to one fold of the cross-validation procedure.  
The sheet **Media_fold** reports the mean values of all performance metrics computed across the five folds.

### Reported metrics

For each experimental treatment, the following columns are provided:

- **Trattamento**: identifier of the experimental treatment or condition  
- **Species Accuracy**: overall classification accuracy  
- **Species Precision**: precision of the species classification  
- **Species Recall**: recall (sensitivity) of the species classification  
- **Species F1**: F1-score of the species classification  
- **Species MCC**: Matthews Correlation Coefficient for species classification  

All metrics are dimensionless and range between 0 and 1, with higher values indicating better classification performance.

---

## Relation to the associated publication

The dataset and the code support the results presented in the associated scientific article and were used to generate the performance analyses reported in the Results section.

---

