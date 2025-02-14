# ECG Chagas

## Introduction

### Overview of the disease

### About the challenge

### Dataset

* balanced on purpose for the sake of the project to avoid dealing with imbalanced data

### Example of the ECG signal

### ECG leads waves, segments and intervals

### Feature extraction and data processing

* sampling rate
* null values
* how waves are extracted and R peaks computed

## EDA

### Distribution of the target variable

### Distribution of the features

### Welch’s t-Test

### Distribution of categorical columns

### Target variable distribution by gender

### Distribution of numerical columns conditioned on the target variable

### Correlation matrix
* slightly positive correlation between the target variable and the features

### Age divided in 4 groups with qcut
* the 4 groups in this way are quartiles

### Ventricular Rate

### Heart Rate Variability (HRV_HTI)

### Heart Rate Variability (HRV_SDSD)

### QRS Duration

### PR duration

### PR Interval with literature values

### Standardization of data

### Results function and metrics

## Logistic Regression

### General idea

### Loss function with L1 regularization
* L1 regularization is used to shrink the coefficients of the features that are not important for the model

### C hyperparameter discussion
* C is the inverse of the regularization strength, so the smaller the C the stronger the regularization

### GridSearchCV and results
* model selects the best C value for the model to be 10 thus promoting less regularization. This suggests that the model does not require much regularization to be effective in Cross-Validation giving insights into the fact that the problem is not too complex.

### ROC-AUC score as a metric
* what measures?

### Coefficients interpretation

### Coefficient plot

## XGBoost

### Why do we need to use XGBoost instead of Logistic Regression?

### General idea

### Optimal hyperparameters

### Results function and metrics

### Feature importance

### All feature importances

## Support Vector Classifier

### Hyperparameters via Optuna

### Results function and metrics

## K-Nearest Neighbors

### Why another baseline model?

### Binary search over n_neighbors

### Results function and metrics

## Permutation Importance

### Why not use feature importance?

### Difference between permutation importance on train and test to assess overfitting

## SHAP Values

### General idea

### SHAP Values interpretation

### Summary plot

### waterfall_plot (negative)

### waterfall_plot (positive)

### beeswarm plot

### Feature selection with top 20 SHAP features

### Results

### BorutaPy

### Results

### Intersection of SHAP and Boruta models results

### McNemar's test on SHAP and Boruta results checking if they are statistically different

### Final Classic ML model summary

## Deep Learning

### Why?

### Explainability

### Avoid artifacts by truncating the signal to minimum length

### Model architecture

### Hyperparameters

### Results

### Grad-CAM

## Transformer Models

### General idea

### Feature extraction

### Positional encoding

### Hyperparameters

### Results

### Adding demographic features

### Results

### Adding the top 20 SHAP features along with demographic features

### Results

### Final model comparison

## References
