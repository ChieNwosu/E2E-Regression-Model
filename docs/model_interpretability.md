# Model Interpretability

## Approach

Linear Regression was selected as the final model. Because all features were standardized using `StandardScaler` before training, the model's coefficients are directly comparable across features.

**Interpretation rule:** Each coefficient represents the change in predicted house price (in USD) for a **one standard deviation increase** in that feature, holding all other features constant.

---

## What Standardized Coefficients Mean

| Coefficient Sign | Meaning |
|---|---|
| Positive | Increasing this feature increases predicted price |
| Negative | Increasing this feature decreases predicted price |
| Larger absolute value | Stronger influence on predicted price |

---

## Top Features by Coefficient Magnitude

Based on the trained Linear Regression model (features standardized):

**Strongest positive drivers of price:**
- `grade` — construction and design quality; higher grade strongly increases price
- `sqft_living` — interior living area; larger homes command higher prices
- `view` — quality of view rating; premium views add significant value
- `waterfront` — waterfront access; rare but high-impact feature
- `sqft_living15` — neighborhood living area average; reflects desirable areas

**Strongest negative drivers of price:**
- `zipcode` — treated as a raw numeric feature in this model; the negative coefficient reflects the inverse numeric encoding of certain zip codes, not a true causal relationship
- `long` — longitude; reflects geographic pricing patterns in the King County area
- `yr_built` — older homes tend to have lower prices, all else equal

> **Note:** `zipcode` and `long` are location proxies. Their coefficients should be interpreted cautiously — they encode geographic patterns, not direct causal relationships. A better approach would be to encode `zipcode` categorically or use regional price aggregates.

---

## Coefficient Chart

See `visuals/coefficient_importance.png` for a bar chart of all feature coefficients sorted by magnitude.

---

## Limitations of Coefficient-Based Interpretability

- Coefficients assume **linear relationships** — if the true relationship is non-linear, the coefficient is a linear approximation
- **Correlated features** can split their true effect across multiple coefficients, making individual coefficients harder to interpret
- `zipcode` as a raw numeric feature is a known limitation — it should be treated as categorical
- This approach does not capture **interaction effects** between features

---

## Alternative Interpretability Methods (Future Work)

For more robust interpretability, consider:
- **SHAP values** — model-agnostic, handles non-linear models
- **Partial Dependence Plots (PDPs)** — visualize marginal effect of each feature
- **Permutation importance** — measures how much performance drops when a feature is shuffled
