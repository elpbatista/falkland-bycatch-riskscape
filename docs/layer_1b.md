# Note: Layer 1b — Dynamic Seascape Classification (Future Work)

## Context

Layer 1 currently represents the **continuous physical ocean state** projected to the H3 grid:

- SST (sea surface temperature)
- Chlorophyll-a
- SSH (sea surface height)

These variables are stored as continuous predictors and serve as the foundation for:

- Gradient computation (Layer 2)
- Anomaly detection
- ML-based bycatch risk modeling

---

## Question

Should we implement a classified dynamic seascape layer (Layer 1b), similar to María’s work?

---

## Concept

Layer 1b would consist of a **daily categorical regime map**, derived by clustering multivariate physical fields.

Structure:

```text
date | h3 | regime_id
```

Where:

- `regime_id` ∈ {1, …, k}
- k ≈ 5–10 dynamic ocean regimes

This would generate:

~3,650 daily classified maps (10 years × daily resolution)

But only k distinct regime types, shifting spatially through time.

---

## Implementation Strategy (To Revisit Later)

Preferred approach:

- Global clustering across all days
- Fixed regime definitions
- Daily spatial assignment

Not recommended:

- Independent per-day clustering (regime meaning becomes inconsistent)

---

## Scientific Rationale

Continuous fields (Layer 1) preserve full physical gradients.

Classification (Layer 1b):

- Reduces dimensionality
- Increases interpretability
- Aligns with NOAA-style dynamic seascapes
- Facilitates communication with managers

However:

Continuous gradients (Layer 2) may be more predictive for seabird foraging and bycatch risk.

---

## Strategic Decision

Layer 1b will be considered **after completing Layer 2 (derived predictors)**.

Planned evaluation:

1. Train baseline model with continuous predictors.
2. Add regime classification as an additional predictor.
3. Compare performance and interpretability.

Research question:

> Does discrete dynamic seascape classification improve bycatch risk prediction relative to continuous physical predictors?

---

## Status

Deferred until:

- Layer 2 gradients and anomalies are implemented
- Baseline model is established

This note serves as a reminder to revisit dynamic seascape classification after physical feature engineering is complete.
