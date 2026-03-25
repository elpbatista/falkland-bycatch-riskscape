# Layer 1b — Dynamic Seascape Classification (Implementation Note)

## Context

Layer 1 represents the continuous physical ocean state projected to the H3 grid:

- SST (sea surface temperature)
- Chlorophyll-a (CHL)
- SSH (sea surface height)

These variables are derived from satellite and reanalysis products and are used as inputs for:

- Gradient computation (Layer 2)
- Anomaly detection
- Machine learning models for bycatch risk

---

## Key Finding — CHL Missingness

Raw data diagnostics (CMEMS L4 CHL product) showed:

- CHL contains missing values **every day**
- Missing fraction varies over time:
  - Minimum: ~1%
  - Median: ~1%
  - Maximum: ~28%

Conclusion:

> CHL is **not spatially complete**, and missingness is **dynamic (time-dependent)** rather than constant.

---

## Implication

CHL cannot be used to define spatial validity.

If used as a strict filter:

- Up to ~28% of ocean cells would be removed on some days
- This introduces spatial bias and artificial gaps

---

## Design Decision

Adopt the following strategy:

### Training

Use only rows where all variables are valid:

- SST: finite
- CHL: finite
- SSH: finite

This ensures:

- Physically consistent clustering
- No artificial values influence regime definition

---

### Prediction

Use all rows where core physical variables exist:

- SST: required
- SSH: required
- CHL: optional

For rows where CHL is missing:

- Impute CHL using the median (within the prediction batch)
- Apply trained scaler and clustering model

This ensures:

- Full spatial coverage
- No loss of ocean areas due to CHL gaps

---

## Conceptual Model

Variables are treated as:

### Core variables (define structure)

- SST
- SSH

### Auxiliary variable (enhances regimes)

- CHL

CHL contributes to regime differentiation but does not define where regimes exist.

---

## Workflow Summary

1. Load Layer 1 data (SST, CHL, SSH)
2. Apply log transform to CHL
3. Build feature matrix

### Training phase

1. Select rows where SST, CHL, SSH are all finite
2. Sample subset (for scalability)
3. Fit:
   - StandardScaler
   - KMeans (k ≈ 5–10)

### Prediction phase

1. Select rows where SST and SSH are finite
2. Impute CHL where missing
3. Transform using fitted scaler
4. Predict regime_id

### Output

- Structure: `date | h3 | regime_id`
- Saved by year

---

## Rationale

This approach:

- Preserves physical realism
- Avoids bias from missing CHL
- Maintains full spatial coverage
- Aligns with observed data limitations

---

## Status

Layer 1b implemented with:

- k = 7 (initial choice)
- Sampling-based training
- Robust handling of CHL missingness

---

## Future Work

- Evaluate sensitivity to k (e.g., 5–10)
- Characterize regimes (mean SST, CHL, SSH)
- Compare models:
  - With vs without regimes
- Assess contribution of regimes to bycatch risk prediction
