# Project Overview: End-to-End Machine Learning Regression — House Price Prediction

## Summary

This project builds a complete machine learning regression pipeline to predict residential house sale prices using structured property features. It covers the full ML lifecycle: exploratory data analysis, preprocessing, model training, cross-validation, evaluation, and interpretability.

## Learning Context

This project was completed while working through **"Mastering ChatGPT and Google Colab for Machine Learning"** by Moscato. The original notebook follows a chapter-based structure:

- **Chapter 9** — Data preparation, EDA, feature analysis, geospatial visualization
- **Chapter 10** — Model fine-tuning, cross-validation, model selection, interpretability

The notebook is preserved as a learning artifact. A cleaned portfolio version is available in `notebooks/E2E_ML_Regression_Model_v2_clean.ipynb`.

## Problem Type

| Attribute | Value |
|---|---|
| Task | Supervised Regression |
| Target Variable | `price` (continuous, USD) |
| Domain | Real estate / housing market |
| Dataset Size | 21,613 records × 21 columns |
| Missing Values | 0 |

## Dataset

The dataset contains King County, WA residential property sale records. Key features include:

- **Size:** `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`
- **Structure:** `bedrooms`, `bathrooms`, `floors`
- **Quality:** `grade`, `condition`, `view`, `waterfront`
- **Age:** `yr_built`, `yr_renovated`
- **Location:** `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`

## Workflow Summary

```
Raw CSV → EDA → Feature Drop → Train/Test Split (70/30)
       → StandardScaler (fit on train only)
       → Linear / Ridge / Lasso Regression
       → 5-Fold Cross-Validation
       → Test Set Evaluation (MAE, RMSE, R²)
       → Model Selection → Coefficient Interpretability
```

## Models Compared

| Model | Regularization |
|---|---|
| Linear Regression | None |
| Ridge Regression | L2 |
| Lasso Regression | L1 |

## Final Model

**Linear Regression** — selected because all three models performed identically, and Linear Regression offers the most interpretable coefficients.

## Files

| File | Description |
|---|---|
| `ML_Regression_Analysis_Model_github.ipynb` | Original chapter-based learning notebook |
| `notebooks/E2E_ML_Regression_Model_v2_clean.ipynb` | Cleaned portfolio notebook |
| `House_Prices.csv` | Original dataset |
| `data/raw/House_Prices.csv` | Dataset copy for v2 notebook |
| `visuals/` | Generated charts |
| `docs/` | Documentation files |
