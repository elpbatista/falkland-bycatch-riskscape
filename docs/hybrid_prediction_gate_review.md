# Hybrid Prediction Gate Review

Last verified: 2026-05-11 15:32:53 PDT

This note documents the current hybrid species-use prediction equation before rerunning prediction surfaces. It is intended to prevent accidental changes to the gating logic or accidental use of outdated model payloads.

## Current Code Paths

Prediction entry point:

`Repo/scripts/predict_models.py`

Prediction implementation:

`Repo/src/riskscape/model/predict.py`

Relevant functions:

- `build_hybrid_product_predictions`
- `build_product_predictions`
- `build_product_plausibility`
- `presence_gate`
- `apply_presence_gate`
- `add_risk_columns`

## Current Hybrid Mode

The current prediction mode is:

```python
PREDICTION_MODE = "hybrid"
HYBRID_MODE = "presence_gate"
HYBRID_ML_MODEL_NAME = "extra_trees"
HYBRID_BAYESIAN_MODEL_NAME = "bayesian_gmm"
```

The output folder name is therefore:

`hybrid_presence_gate_extra_trees_bayesian_gmm`

## Current Gate Equation

Let:

- \(x_{h,t}\) be the environmental/static feature vector for H3 cell \(h\) and day \(t\).
- \(s\) be species.
- \(m_s(h,t)\) be the Extra Trees prediction on the log-transformed residence-index scale.
- \(d_s(h,t)\) be the Bayesian/GMM log density for the same species-cell-day.
- \(d_{\min}\) and \(d_{\max}\) be the GMM model's stored 1st and 99th percentile log-density bounds from positive-use training rows.
- \(p_s(h,t)\) be normalized environmental plausibility.
- \(c_s\) be the species-specific maximum cut applied when plausibility is zero.

The current plausibility normalization is:

\[
p_s(h,t) =
\mathrm{clip}
\left(
\frac{d_s(h,t)-d_{\min}}{d_{\max}-d_{\min}},
0,
1
\right)
\]

The current species-specific gate is:

\[
g_s(h,t) = 1 - c_s \left(1 - p_s(h,t)\right)
\]

Current configured maximum cuts are:

```python
HYBRID_GATE_MAX_CUTS = {
    "BBAL": 0.1,
    "SAFS": 0.1,
}
```

Therefore, with the current settings:

\[
g_s(h,t) \in [0.9, 1.0]
\]

This is a soft gate. Even when plausibility is zero, the species-use prediction is reduced by only 10%, not removed.

The Extra Trees prediction is converted from log space to raw space before gating:

\[
u_s(h,t) = \exp(m_s(h,t)) - 1
\]

The gated raw species-use estimate is:

\[
u^{*}_s(h,t) = u_s(h,t) \cdot g_s(h,t)
\]

The output species-use prediction is converted back to log space:

\[
m^{*}_s(h,t) = \log(1 + u^{*}_s(h,t))
\]

The prediction output column `species_use_log_pred` stores \(m^{*}_s(h,t)\). The original ungated Extra Trees prediction is retained as `species_use_ml_log_pred`.

## Current Risk Equation

Fishing effort is stored as:

\[
f(h,t) = \log(1 + F(h,t))
\]

where \(F(h,t)\) is fishing activity in raw vessel-hours.

The current risk score is:

\[
r_s(h,t) = m^{*}_s(h,t) + f(h,t)
\]

Equivalently:

\[
r_s(h,t) =
\log(1 + u^{*}_s(h,t)) +
\log(1 + F(h,t))
\]

This is a log-space interaction between gated species use and fishing effort. It is not currently computed as raw \(u^{*}_s(h,t) \times F(h,t)\), although it is monotonic in both terms.

## Role of K = 15 k-means Seascapes

The K = 15 k-means seascape layer is currently not part of the prediction equation.

Its role so far is validation and interpretation:

- It is the selected ecological BlockCV-style validation split.
- It provides interpretable environmental regimes for reporting.
- It was used to verify that Extra Trees remains the strongest model under environmental blocking.

Current authoritative validation comparison:

`Repo/data/modeling/metrics/species_model_random12_vs_kmeans_k15_block_cv_comparison.csv`

If K = 15 seascapes are intended to affect predictions directly, that would require a new explicitly defined equation. The current hybrid prediction code does not use seascape labels as inputs, weights, gates, or masks.

## Model Payload Issue Before Rerunning Predictions

The current prediction code loads:

- `Repo/data/modeling/models/extra_trees/species_model_joint.joblib`
- `Repo/data/modeling/models/bayesian_gmm/species_model_joint.joblib`

Important verification:

- The current default `bayesian_gmm/species_model_joint.joblib` payload has 10 GMM components.
- The selected 30-component Bayesian/GMM payload exists at:

`Repo/data/modeling/models/bayesian_gmm_component_sweep_random12_selected/components=30/species_model_joint.joblib`

Therefore, rerunning predictions without changing model paths would still use the old/default 10-component Bayesian/GMM gate. Before rerunning final hybrid predictions, the code must either:

1. copy or promote the selected K = 30 Bayesian/GMM payload into the production `bayesian_gmm` model folder, or
2. update prediction code to accept explicit ML and Bayesian model payload paths/output names.

Option 2 is safer because it avoids overwriting older production products and keeps the new run clearly labeled.

## Recommended Next Implementation Step

Before generating new prediction surfaces:

1. Add explicit CLI/config options in prediction code for:
   - ML payload path,
   - Bayesian/GMM payload path,
   - output model name.
2. Run the hybrid prediction with:
   - Extra Trees as the ML species-use model,
   - selected Bayesian/GMM K = 30 as the plausibility gate,
   - output label that explicitly records the new gate, for example:

`hybrid_presence_gate_extra_trees_bayesian_gmm_k30`

3. Keep K = 15 seascapes as the validation/reporting layer unless a new direct seascape-conditioned prediction equation is deliberately defined.
