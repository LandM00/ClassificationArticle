# ClassificationArticle

1) DatasetLight.zip represents a light version of the full experimental dataset used in the associated study. It contains a representative subset of the original images and is provided to support transparency, reproducibility, and methodological understanding, while limiting data volume.
The light dataset includes RGB images of four plant species acquired under controlled conditions and different light treatments. The images were selected to preserve the diversity of species, growth stages, and illumination conditions present in the full dataset.
The dataset is organized according to the following hierarchical structure:
    - Species
      - Growth stage
        - Light treatment
          - Image files (.jpg)
Where:
Species ∈ {Aubergine, Basil, Cucumber, Tomato};
Growth stage ∈ {GerminationStage, VegetativeStage};
Light treatment ∈ {RB1, RB3, RB5, RB7, RB9}.
Each image belongs to exactly one species, one growth stage, and one light treatment.
Images were acquired in controlled experimental conditions using a fixed imaging setup. All images are provided in JPEG format. During model training, images were resized to 256 × 192 pixels and normalized to the [0, 1] range.

2) ClassificationAlgo_def.py is the Python script that implements the training and evaluation pipeline of a multi-task convolutional neural network (CNN) for the simultaneous classification of plant species and growth stages in the presence of different light treatments. The code performs data loading, preprocessing, model training, k-fold cross-validation, performance metric calculation, confusion matrix generation, and export of final results to Excel files and image graphs.
 
3) Metrics_CNN_results.xlsx contains the performance metrics obtained from the evaluation of a Convolutional Neural Network (CNN) model for plant species classification under different experimental treatments. The dataset reports the results of a five-fold cross-validation procedure, together with the average performance across folds, and was used to support the quantitative analysis presented in the associated scientific article.
The Excel file is organized into six worksheets: Fold_1, Fold_2, Fold_3, Fold_4, Fold_5, Media_fold.
Each worksheet corresponds to one fold of the cross-validation procedure. The sheet Media_fold reports the mean values of all performance metrics computed across the five folds.
For each experimental treatment, the following columns are provided:
  - Trattamento: identifier of the experimental treatment or condition.
  - Species Accuracy: overall classification accuracy of the CNN model.
  - Species Precision: precision of the species classification.
  - Species Recall: recall (sensitivity) of the species classification.
  - Species F1: F1-score of the species classification.
  - Species MCC: Matthews Correlation Coefficient for species classification.

All metrics are dimensionless and range between 0 and 1, with higher values indicating better classification performance.
This dataset supports the results presented in the associated article and was used to generate the performance analyses reported in the Results section.
