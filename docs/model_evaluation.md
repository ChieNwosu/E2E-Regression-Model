# Model Evaluation

## Overview

Three regression models were evaluated using a held-out test set (30% of data) and 5-fold cross-validation on the training set. All metrics were computed on standardized features.

**Split:** 70% train / 30% test | `random_state=42`  
**Scaling:** StandardScaler fit on training data only

---

## Test Set Results

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | $127,630 | $43,413,626,657 | $208,359 | 0.6993 |
| Ridge Regression | $127,628 | $43,413,792,577 | $208,360 | 0.6993 |
| Lasso Regression | $127,631 | $43,413,760,727 | $208,360 | 0.6993 |

---

## 5-Fold Cross-Validation Results (Training Set)

| Model | CV MAE (mean) | CV MAE (std) | CV R² (mean) | CV R² (std) |
|---|---|---|---|---|
| Linear Regression | $125,059 | $515 | 0.6977 | 0.0112 |
| Ridge Regression | $125,056 | $515 | 0.6977 | 0.0112 |
| Lasso Regression | $125,059 | $515 | 0.6977 | 0.0112 |

---

## Metric Definitions

| Metric | Definition | Interpretation |
|---|---|---|
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual price | Lower is better; in same units as price (USD) |
| **MSE** | Mean Squared Error — average squared difference | Penalizes large errors more heavily; sensitive to outliers |
| **RMSE** | Root Mean Squared Error — square root of MSE | Same units as price; easier to interpret than MSE |
| **R²** | Coefficient of Determination — proportion of variance explained | 1.0 = perfect; 0.0 = no better than predicting the mean |

---

## Interpretation

- **R² ≈ 0.70** means the model explains approximately 70% of the variance in house prices. The remaining 30% is unexplained — likely due to non-linear relationships, missing features, or noise.
- **MAE ~$127K** means the average prediction error is about $127,000. In context of the price range ($75K–$7.7M), this is a reasonable baseline but leaves room for improvement.
- **Small CV standard deviations** (MAE std ~$515, R² std ~0.011) confirm the model is stable and generalizes consistently across folds.
- **MSE is large** because squared errors amplify the effect of high-value outliers (homes priced $2M–$7.7M).
- **All three models perform identically** — regularization at default `alpha=1.0` did not improve performance, indicating the feature set is already well-conditioned after standardization.

---

## Why Linear Regression Was Selected

1. Performance is statistically equivalent to Ridge and Lasso
2. Coefficients are directly interpretable (see `docs/model_interpretability.md`)
3. No hyperparameter tuning required for equivalent results
4. Serves as a transparent, explainable baseline

---

## Visuals

- `visuals/actual_vs_predicted.png` — scatter plot of actual vs. predicted prices
- `visuals/residual_plot.png` — residuals vs. predicted and residual distribution
- `visuals/model_comparison_metrics.png` — bar chart comparing MAE, RMSE, R² across models

---

## Notes on Metrics Reported in Original Notebook

The original notebook (`ML_Regression_Analysis_Model_github.ipynb`) reports the same metrics with consistent values:
- Linear Regression MAE: ~127,630 | R²: 0.70
- Ridge Regression MAE: ~127,628 | R²: 0.70
- Lasso Regression MAE: ~127,631 | R²: 0.70

The v2 notebook adds RMSE and uses a cleaner preprocessing pipeline (scaler fit on train only), but produces the same results because the original notebook also fit the scaler on the full dataset before splitting — a minor methodological difference that did not materially affect outcomes on this dataset.
