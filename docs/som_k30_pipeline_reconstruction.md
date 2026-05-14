# SOM k30 Pipeline Reconstruction

This document records the commands needed to rebuild the current data-informed
pipeline using the SOM-hierarchical k=30 seascape validation design.

Run commands from the repository root:

```bash
cd /Users/pb/Work/OSU/CapstoneProject/Repo
```

Use `PYTHONPATH=src` when running scripts from inside `Repo`.

## Selected Model Identity

Selected validation design:

```text
SOM-hierarchical k=30 + 5-fold grouped environmental BlockCV
```

Production species-use model name:

```text
extra_trees_som_hierarchical_k30_5fold_blockcv
```

Hybrid prediction product name:

```text
hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30
```

## 1. Rebuild SOM k30 Seascape Assignments

This reuses the fitted 15 x 15 SOM and exports the hierarchical k=30 cut.

```bash
PYTHONPATH=src python3 scripts/classify_som_hierarchical_seascapes.py \
  --years 2014-2023 \
  --source-model-name som_15x15_hierarchical_k15 \
  --model-name som_15x15_hierarchical_k30 \
  --n-classes 30 \
  --export-cut
```

Expected output root:

```text
data/modeling/seascapes/som_15x15_hierarchical_k30/
```

## 2. Rebuild Selected BlockCV Evidence

```bash
PYTHONPATH=src python3 scripts/run_block_cv_variant_comparison.py \
  --variants selected \
  --out-prefix species_model_block_cv_selected_som_k30_5fold
```

Expected metric output:

```text
data/modeling/metrics/species_model_block_cv_selected_som_k30_5fold.csv
```

## 3. Train Production Species-Use Model

```bash
PYTHONPATH=src python3 scripts/train_active_species_model.py
```

Expected model output:

```text
data/modeling/models/extra_trees_som_hierarchical_k30_5fold_blockcv/species_model_joint.joblib
```

Expected metric output:

```text
data/modeling/metrics/species_model_extra_trees_som_hierarchical_k30_5fold_blockcv_production_metrics.csv
```

## 4. Run Hybrid Predictions

This is the heaviest step.

```bash
PYTHONPATH=src python3 scripts/predict_models.py
```

Expected prediction output:

```text
data/modeling/predictions/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/joint/
```

## 5. Rebuild Prediction Maps

```bash
PYTHONPATH=src python3 scripts/plot_prediction_maps.py
```

## 6. Rebuild Monthly Latent-Risk Matrix

```bash
PYTHONPATH=src python3 scripts/plot_prediction_latent_risk_monthly_matrix.py \
  --color-bin-source monthly_species \
  --color-quantiles 0 0.40 0.75 0.95 1.0
```

## 7. Rebuild Weekly Operator Products

Build weekly climatology and 2022 sequence tables:

```bash
PYTHONPATH=src python3 scripts/build_weekly_operator_latent_risk.py
```

Plot small multiples and animation:

```bash
PYTHONPATH=src python3 scripts/plot_weekly_operator_latent_risk.py \
  --make-small-multiples \
  --make-animation
```

Plot fisheries-grid example:

```bash
PYTHONPATH=src python3 scripts/plot_weekly_operator_fisheries_grid_example.py
```

## 8. Optional Seascape-Risk Experiment

Build SOM k30 seascape species-use surfaces:

```bash
PYTHONPATH=src python3 scripts/build_seascape_species_use_surfaces.py \
  --years 2014-2023
```

Build the standard seascape-conditioned prediction product for 2022:

```bash
PYTHONPATH=src python3 scripts/plot_seascape_prediction_maps.py \
  --year 2022 \
  --model-name som_15x15_hierarchical_k30 \
  --skip-build
```

If the standard seascape-conditioned prediction product does not exist yet,
omit `--skip-build`:

```bash
PYTHONPATH=src python3 scripts/plot_seascape_prediction_maps.py \
  --year 2022 \
  --model-name som_15x15_hierarchical_k30
```

Then plot the final 2022 seascape latent-risk monthly matrices with the same
layout, color scale, and binned colorbar used by the prediction risk matrices:

```bash
PYTHONPATH=src python3 scripts/plot_prediction_latent_risk_monthly_matrix.py \
  --model-name seascape_som_15x15_hierarchical_k30 \
  --product-name joint \
  --year 2022 \
  --agg non_zero_mean \
  --color-bin-source monthly_species \
  --color-quantiles 0 0.40 0.75 0.95 1.0
```

Expected final seascape-risk matrix outputs:

```text
plots/predictions/seascape_som_15x15_hierarchical_k30_joint_latent_risk_log_pred_non_zero_mean_BBAL_2022_monthly_matrix.png
plots/predictions/seascape_som_15x15_hierarchical_k30_joint_latent_risk_log_pred_non_zero_mean_SAFS_2022_monthly_matrix.png
```

Do not use the `--monthly-matrix` mode in `plot_seascape_prediction_maps.py`
for final seascape-risk matrices. That path is useful for quick exploratory
diagnostics, but it uses a different matrix layout and colorbar logic:

```bash
PYTHONPATH=src python3 scripts/plot_seascape_prediction_maps.py \
  --year 2022 \
  --model-name som_15x15_hierarchical_k30 \
  --monthly-matrix \
  --matrix-values risk_log_pred \
  --agg non_zero_mean
```

## Notes

- Do not run scripts directly as executables, for example
  `scripts/plot_weekly_operator_latent_risk.py`; use
  `python3 scripts/plot_weekly_operator_latent_risk.py`.
- The old k-means k=15 products are not overwritten by the new hybrid product
  name.
- Animation frames and videos are ignored by git under
  `plots/predictions/weekly_operator/`.
- The SOM k30 choice is data-informed by grouped BlockCV performance, but it
  should still be described with the known interpretability/species-support
  tradeoff.
