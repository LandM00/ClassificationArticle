Optimizing artificial lighting for convolutional neural network-based crop monitoring with low-cost RGB imaging in indoor cultivation (doi.org/10.1016/j.atech.2025.101677)

The repository includes the following main components:
- DatasetLight.zip – light version of the image dataset;
- ClassificatoreAlgo_def.txt – Python training and evaluation script;
- Metrics_CNN_results.csv – performance metrics from cross-validation.

DatasetLight.zip
	DatasetLight.zip represents a light version of the full experimental dataset used in the associated study. It contains a representative subset of the original images and is 	provided to support transparency, reproducibility, and methodological understanding, while limiting data volume.
	The light dataset includes RGB images of four plant species acquired under controlled conditions and different light treatments. The images were selected to preserve the diversity 	of species, growth stages, and illumination conditions present in the full dataset.
	The dataset is organized according to the following hierarchical structure:
		- Species
			- Growth stage
				- Light treatment
					- Image files (.jpg)
	Where:
		- Species ∈ {Aubergine, Basil, Cucumber, Tomato}
		- Growth stage ∈ {GerminationStage, VegetativeStage}
		- Light treatment ∈ {RB1, RB3, RB5, RB7, RB9}
	Each image belongs to exactly one species, one growth stage, and one light treatment.
	Images were acquired in controlled experimental conditions using a fixed imaging setup.
	All images are provided in JPEG format.
	During model training, images were resized to 256 × 192 pixels and normalized to the [0, 1] range.

ClassificatoreAlgo_def.txt
	ClassificatoreAlgo_def.txt is the Python script that implements the training and evaluation pipeline of a multi-task convolutional neural network (CNN) for the simultaneous 	classification of plant species and growth stages in the presence of different light treatments.
	The code performs the following main steps:
		- data loading and labeling
		- image preprocessing and data augmentation
		- definition of a multi-task CNN architecture
		- five-fold cross-validation training
		- performance metric computation
		- confusion matrix generation
		- export of results to Excel files and image plots
	The code is written in Python and relies on the following main libraries:
		- TensorFlow / Keras
		- NumPy
		- scikit-learn
		- Pandas
		- Matplotlib
	GPU acceleration and mixed-precision training are supported.

Metrics_CNN_results.csv
	Metrics_CNN_results.csv contains the performance metrics obtained from the evaluation of the CNN model for the simultaneous classification of plant species and growth stage under different 	experimental treatments.
	The dataset reports the results of a five-fold cross-validation procedure, together with the average performance across folds, and was used to support the quantitative analysis presented in the 	associated scientific article.
	The file is provided in CSV format in order to ensure interoperability and long-term accessibility, in compliance with FAIR data principles.
	The CSV file is organized in tabular form, where each row corresponds to a specific experimental treatment and cross-validation fold.
	
	The dataset includes both:
	- metrics computed for each individual fold,
	- mean values computed across the five folds.
	- A dedicated column indicates the fold number, with the value AVG used to denote the average across folds.

	For each experimental treatment, the following columns are provided:
	-Fold: cross-validation fold identifier (1–5) or AVG for the average across folds
	- Trattamento: identifier of the experimental treatment or condition
	
	Species classification metrics:
	- Species Accuracy: overall species classification accuracy
	- Species Precision: precision of the species classification
	- Species Recall: recall (sensitivity) of the species classification
	- Species F1: F1-score of the species classification
	- Species MCC: Matthews Correlation Coefficient for species classification

	Growth stage classification metrics:
	- Stage Accuracy: overall growth stage classification accuracy
	- Stage Precision: precision of the growth stage classification
	- Stage Recall: recall (sensitivity) of the growth stage classification
	- Stage F1: F1-score of the growth stage classification
	- Stage MCC: Matthews Correlation Coefficient for growth stage classification
	- All metrics are dimensionless and range between 0 and 1, with higher values indicating better classification performance.

This dataset supports the results presented in the associated scientific article and was used to generate the performance analyses reported in the Results section.