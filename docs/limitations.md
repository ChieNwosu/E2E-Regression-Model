# Limitations

## Model Limitations

### 1. Linear Relationship Assumption
Linear Regression assumes a linear relationship between each feature and the target variable. Housing prices often exhibit non-linear patterns — for example, the price premium for an extra bedroom diminishes as homes get larger. Non-linear models (Random Forest, Gradient Boosting) may capture these patterns better.

### 2. Heteroscedasticity
The residual plot shows increasing variance at higher predicted prices. This means the model is less accurate for expensive homes than for mid-range homes. A log transformation of the target variable (`log(price)`) could stabilize variance and improve performance at the high end.

### 3. Outlier Sensitivity
MSE is disproportionately affected by high-value outliers (homes priced $2M–$7.7M). These extreme values inflate the squared error metric without necessarily indicating poor model performance on typical homes.

### 4. Regularization Not Tuned
Ridge and Lasso were evaluated at default `alpha=1.0`. Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV` may reveal settings where regularization provides meaningful improvement.

## Data Limitations

### 5. Zipcode as Numeric Feature
`zipcode` is treated as a raw integer in this model. Zip codes are categorical identifiers — their numeric ordering is arbitrary and does not reflect geographic proximity or price relationships. This should be encoded as a categorical variable or replaced with regional price aggregates.

### 6. Dropped Features
`yr_renovated` was dropped because ~96% of values are 0 (no renovation recorded). A binary `was_renovated` feature or a `years_since_renovation` feature might recover some signal.

`date` was dropped entirely. Sale date could encode seasonal pricing patterns or market trends that affect price.

### 7. No Temporal Validation
The model was not tested for temporal generalization. If trained on older sales and tested on newer ones, performance may differ due to market changes. A time-based split would be more realistic than a random split.

### 8. Location Encoding
`lat` and `long` are included as raw numeric features. While they carry location signal, a more effective approach would be to cluster properties by neighborhood or use distance-to-amenity features.

## Scope Limitations

### 9. No Production Deployment
This is a portfolio and learning project. The model has not been deployed to a production environment, integrated into an API, or tested for real-time inference at scale.

### 10. No Advanced MLOps
There is no model versioning, experiment tracking (MLflow), or automated retraining pipeline. These would be necessary for a production system.

### 11. Single Dataset
The model was trained and evaluated on a single dataset from one geographic region (King County, WA). Generalization to other markets is not validated.

## What These Limitations Mean in Practice

An R² of 0.70 and MAE of ~$127K is a reasonable baseline for a linear model on this dataset. However, these limitations suggest that:
- A non-linear model with proper feature engineering could meaningfully improve performance
- The model should not be used for high-stakes pricing decisions without further validation
- Results should be interpreted as a learning demonstration, not a production-ready system
