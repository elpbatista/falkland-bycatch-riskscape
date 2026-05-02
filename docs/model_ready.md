# Layer 2 → Model Ready Dataset (Final Validated Specification)

This document defines the construction, validation, and final state of the environmental feature space used for modeling. It represents the complete and validated output of Layer 2 and the standardized `model_ready` dataset.

---

## 1. Architecture Overview

The data pipeline follows a three-layer structure:

Layer 1 → Layer 2 → model_ready

### Layer Definitions

Layer 1  
Raw environmental variables extracted per H3 cell and date.

Layer 2A  
Spatial structure captured via gradients.

Layer 2B  
Temporal variability captured via anomalies.

model_ready  
Standardized predictors ready for modeling.

---

## 2. Layer 1 Variables

Base environmental variables:

- sst — sea surface temperature
- chl — chlorophyll-a concentration (log10-transformed)
- ssh — sea surface height
- wind — wind magnitude

### Chlorophyll Treatment

Chlorophyll is transformed as:

chl = log10(chlorophyll concentration)

Properties:

- No zero values were present
- No epsilon correction was required
- Values are stored in log space permanently

---

## 3. Layer 2A — Spatial Gradients

Spatial gradients quantify local contrast using H3 ring-1 neighbors.

### Definition

For each cell i:

grad(i) = sqrt( mean( (Xi − Xj)^2 ) )

Where:

- j are the 6 neighboring H3 cells
- Only valid neighbors are used
- Missing values are ignored

### Variables

- sst_grad
- chl_grad (computed in log space)
- ssh_grad
- wind_grad

### Interpretation

- High values indicate strong fronts or spatial transitions
- Low values indicate homogeneous regions

---

## 4. Layer 2B — Temporal Anomalies

Anomalies quantify deviation from seasonal climatology.

## Climatology Definition

For each H3 cell i and day-of-year d:

X̄(i,d) = mean over all years of X(i,d)

Computed using full 2014–2023 dataset.

### Anomaly Definition

X_anom(i,t) = X(i,t) − X̄(i,d(t))

Where d(t) is the day-of-year.

### More Variables

- sst_anom
- chl_anom (in log space)
- ssh_anom
- wind_anom

### More Interpretation

- Positive → above seasonal expectation
- Negative → below seasonal expectation

### Chlorophyll Interpretation

Because chl is log-transformed:

- chl_anom represents multiplicative deviation
- Example:
  - +0.3 ≈ ~2× increase
  - -0.3 ≈ ~0.5× decrease

---

## 5. Conceptual Feature Structure

Each variable contributes a different physical signal:

### Magnitude

- sst
- chl
- ssh
- wind

Represents absolute environmental state.

### Spatial Contrast

- sst_grad
- chl_grad
- ssh_grad
- wind_grad

Represents fronts and spatial heterogeneity.

### Temporal Deviation

- sst_anom
- chl_anom
- ssh_anom
- wind_anom

Represents departure from seasonal baseline.

---

## 6. Standardization (model_ready)

All predictor variables are standardized using full-period statistics.

### Formula

z = (x − μ) / σ

Where:

- μ = mean over full dataset (2014–2023)
- σ = standard deviation over full dataset

### Standardized Variables

Each variable receives a `_z` version:

- sst_z
- chl_z
- ssh_z
- wind_z
- sst_grad_z
- chl_grad_z
- ssh_grad_z
- wind_grad_z
- sst_anom_z
- chl_anom_z
- ssh_anom_z
- wind_anom_z

### Important Property

Standardization is computed across the full dataset, not per year.

---

## 7. Validation Results

### Full Dataset Validation

All standardized variables satisfy:

- mean ≈ 0
- std ≈ 1

This confirms correct implementation.

### Per-Year Behavior

Per-year means are not expected to be zero.

Observed patterns:

- Warm years → positive sst_anom_z mean
- Cool years → negative sst_anom_z mean

This reflects real interannual variability and is correct.

---

## 8. Data Quality Checks

### Chlorophyll

- Log transform applied consistently
- No zero values
- Distribution is stable
- Anomalies computed in log space

### Gradients

- Computed using vectorized H3 neighbor indexing
- Missing neighbors handled correctly
- No row inflation or duplication

### Anomalies

- Climatology computed using full 10-year dataset
- DOY alignment correct
- No merge inconsistencies detected

### Standardization

- Computed using streaming statistics
- Applied consistently across all years
- Verified across full dataset

---

## 9. Distribution Characteristics

Some variables show heavy tails:

- chl_grad_z
- wind_grad_z

This is expected:

- Gradients represent rare strong fronts
- Extreme values correspond to real physical events

These are not errors and should not be clipped at this stage.

---

## 10. Missing Data

NaN values are present due to:

- Missing source data
- Edge effects in gradients
- Spatial gaps

Counts are consistent across years and variables.

No abnormal inflation detected.

---

## 11. Final Dataset Structure

Each yearly file:

data/model_ready/year=YYYY.parquet

Contains:

- date
- h3
- raw variables
- gradients
- anomalies
- standardized variables (_z)

---

## 12. Final Status

The dataset is now:

- Physically consistent
- Statistically validated
- Fully standardized
- Ready for modeling

Layer 2 is complete and should be considered frozen.

---

## 13. Next Step

Proceed to Layer 3:

- Integration with fishing effort and catch data
- Target variable definition
- Model design and training
