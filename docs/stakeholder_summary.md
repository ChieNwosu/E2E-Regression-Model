# Stakeholder Summary

## What This Project Does

This project uses machine learning to predict residential house sale prices based on property characteristics such as size, quality, location, and structural features.

Three models were built and compared. The final model — Linear Regression — explains approximately **70% of the variation in house prices** and produces predictions with an average error of about **$127,000**.

---

## Key Findings

### What drives house prices most?

Based on the model's feature analysis:

1. **Construction quality (`grade`)** — Homes with higher design and build quality command significantly higher prices
2. **Living area (`sqft_living`)** — Larger homes are worth more; this is the strongest size-based predictor
3. **View quality** — Homes with premium views carry a measurable price premium
4. **Waterfront access** — Rare but high-impact; waterfront homes are priced substantially higher
5. **Neighborhood context (`sqft_living15`)** — The average size of nearby homes reflects neighborhood desirability

### What does NOT predict price as strongly?

- **Bedroom count alone** — Without considering total living area, bedroom count is a weak predictor
- **Lot size** — Large lots do not consistently translate to higher prices in this dataset

---

## Model Performance in Plain Terms

| Metric | Value | What It Means |
|---|---|---|
| R² | 0.70 | The model explains 70% of why prices vary |
| MAE | ~$127,000 | On average, predictions are off by about $127K |
| Stability | High | Consistent results across 5 validation rounds |

**Context:** House prices in this dataset range from ~$75,000 to $7,700,000. A $127K average error is a reasonable baseline for a linear model, but leaves room for improvement — especially for high-value properties.

---

## What This Model Can and Cannot Do

**Can do:**
- Provide a data-driven baseline estimate for typical residential properties
- Identify which features most strongly influence price
- Support exploratory analysis and market understanding

**Cannot do:**
- Reliably predict prices for luxury or unusual properties (high-end outliers)
- Account for market timing, economic conditions, or recent renovations not in the data
- Replace professional appraisal or domain expertise

---

## Recommended Next Steps

For improved accuracy and business applicability:

1. **Apply log transformation to price** — reduces the impact of outliers and improves model fit
2. **Tune regularization** — test Ridge/Lasso with optimized settings
3. **Try non-linear models** — Random Forest or Gradient Boosting typically outperform linear models on housing data
4. **Improve location encoding** — replace raw `zipcode` with neighborhood-level price averages
5. **Add temporal features** — incorporate sale date to capture seasonal and market trends

---

## Project Context

This is a portfolio and learning project completed as part of a structured machine learning curriculum. It demonstrates the ability to build, evaluate, and interpret regression models — not a production-ready pricing system.
