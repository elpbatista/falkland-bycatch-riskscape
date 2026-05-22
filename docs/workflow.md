# Workflow

This document describes the intended public workflow at a high level. The
script inventory in `docs/script_inventory.md` tracks which entry points are
ready to keep and document.

`config.yaml` is the canonical public workflow configuration. Local secrets and
runtime overrides belong in `.env`. Notebooks may demonstrate and inspect
workflow stages, but scripts and package code are the pipeline authority.

Datasets in `config.yaml` have explicit roles:

- `environmental`: raster/time-varying inputs used to build environmental
  feature tables.
- `fishing_effort`: fishing activity input downloaded from Global Fishing Watch.
- `static`: spatial inputs that do not vary through time, such as bathymetry.

Datasets with `build_lookup: true` receive H3 lookup tables. Datasets with a
`provider` are included in automatic source-data downloads.

The preferred public entry point for generated products is:

```bash
python scripts/run_pipeline.py --stage all
```

`--stage all` starts after reference layers and source data have already been
restored or downloaded. To include external downloads, run:

```bash
python scripts/run_pipeline.py --stage all-with-downloads
```

Individual scripts remain available for debugging, reruns, and targeted stages.

## 1. Restore Reference Layers

Download public reference layers used by maps, spatial overlays, and study-area
setup:

```bash
python scripts/data/download_reference_data.py
```

Downloaded reference files are ignored by Git. See `reference/README.md`.

## 2. Configure Credentials

Set provider credentials outside committed files:

- Earthdata: `~/.netrc`
- Copernicus Marine: `copernicusmarine login`
- CDS: `~/.cdsapirc`
- Global Fishing Watch: local `.env` with `GFW_TOKEN`

See `docs/authentication.md`.

## 3. Download Source Data

Download configured provider-backed datasets:

```bash
python scripts/data/download_data.py
```

To download selected datasets:

```bash
python scripts/data/download_data.py --dataset sst chl ssh wind gfw bathymetry
```

Raw downloads are written under `data/raw/` and are ignored by Git.

Bathymetry is downloaded as a cropped GEBCO subset through CEDA OPeNDAP.

## 4. Build Spatial Framework

Build the common H3 spatial frame and static grid-level features.

Run the grouped stage:

```bash
python scripts/run_pipeline.py --stage spatial
```

Granular scripts:

- `scripts/build/build_grid.py`
- `scripts/build/build_static_features.py`

## 5. Build Lookups and Indices

Build reusable lookup tables that connect environmental rasters, seasons,
neighbors, and other spatial relationships to the project grid.

Current script candidates:

- `scripts/build/build_h3_lookup.py`
- `scripts/build/build_neighbor_table.py`
- `scripts/build/build_neighbor_index_table.py`
- `scripts/build/build_seasonal_lookup.py`

## 6. Build Primary Feature Tables

Align source data to the grid and construct the main feature tables.

Run the grouped stage:

```bash
python scripts/run_pipeline.py --stage features
```

Granular scripts:

- `scripts/build/build_environmental_feature_table.py`
- `scripts/build/build_fishing_effort_feature_table.py`
- `scripts/build/build_species_presence_feature_table.py`

## 7. Build Derived Variables

Create derived physical and ecological variables from the primary features.
These include gradients, anomalies, environmental regimes, and other
model-facing derived fields.

Current script candidates:

- `scripts/build/build_derived_features.py`
- `scripts/build/build_environmental_gradients.py`
- `scripts/build/build_environmental_anomalies.py`
- `scripts/build/build_environmental_regime_table.py`
- `scripts/build/build_seascape_species_use_surfaces.py`

## 8. Inspect and Validate Intermediate Products

Inspection stages are part of the workflow, not an afterthought. They should be
run before treating outputs as model-ready.

Current script candidates:

- `scripts/qa/inspect_columns.py`
- `scripts/qa/feature_qa_summary.py`
- `scripts/qa/quick_validation.py`
- `scripts/qa/run_correlation.py`
- `scripts/qa/run_relationships.py`

Diagnostic plots also support inspection:

- `scripts/plots/plot_environmental_histograms.py`
- `scripts/plots/plot_environmental_correlation_heatmap.py`
- `scripts/plots/plot_environmental_daily_timeseries.py`
- `scripts/plots/plot_environmental_monthly_matrix.py`
- `scripts/plots/plot_environmental_gradient_maps.py`
- `scripts/plots/plot_environmental_single_date_maps.py`
- `scripts/plots/plot_fishing_activity_map.py`
- `scripts/plots/plot_fishing_activity_monthly_matrix.py`
- `scripts/plots/plot_fishing_activity_monthly_timeseries.py`
- `scripts/plots/plot_species_presence_maps.py`
- `scripts/plots/plot_relationship_diagnostics.py`

Weekly and operator-facing products are also part of the plotting surface and
need validation before release:

- `scripts/build/build_weekly_operator_latent_risk.py`
- `scripts/plots/plot_weekly_operator_latent_risk.py`
- `scripts/plots/plot_weekly_operator_fisheries_grid_example.py`
- `scripts/plots/plot_weekly_latent_risk_with_jigger_activity.py`
- `scripts/plots/plot_weekly_gear_aware_risk_examples.py`
- `scripts/plots/plot_weekly_operator_latent_risk.py --make-animation`

## 9. Assemble Model-Ready Datasets

Combine feature tables into model-ready datasets.

Run the grouped stage:

```bash
python scripts/run_pipeline.py --stage model-tables
```

Granular script:

- `scripts/build/build_model_datasets.py`

## 10. Train, Predict, and Evaluate

Modeling scripts currently kept for public workflow documentation are:

```bash
python scripts/run_pipeline.py --stage modeling
```

Granular scripts:

- `scripts/model/train_models.py`
- `scripts/model/predict_models.py`
- `scripts/model/evaluate_models.py`

## 11. Plot Outputs

Generated plots are written under `plots/`. Selected static PNG figures are
tracked in Git because the public notebooks display them directly. Large
animations, intermediate frames, CSV exports, and bulk plot products remain
outside Git and are archived in the Zenodo data bundle or regenerated by the
plot scripts.

Run plot groups with:

```bash
python scripts/plots/plot_all_maps.py --group context
python scripts/plots/plot_all_maps.py --group environmental
python scripts/plots/plot_all_maps.py --group predictions
python scripts/plots/plot_all_maps.py --group seascapes
python scripts/plots/plot_all_maps.py --group weekly
python scripts/plots/plot_all_maps.py --group gear
python scripts/plots/plot_all_maps.py --group videos
```

Use `--group all` to run all grouped plot scripts, or `--list` to inspect which
scripts a group will run.

Reference and static context plots:

- `scripts/plots/plot_study_area_map.py`

Prediction, plausibility, and latent-risk plots:

- `scripts/plots/plot_prediction_maps.py`
- `scripts/plots/plot_prediction_latent_risk_monthly_matrix.py`
- `scripts/plots/plot_plausibility_maps.py`
- `scripts/plots/plot_plausibility_monthly_climatology.py`
- `scripts/plots/plot_plausibility_yearly_timeseries.py`
- `scripts/plots/plot_plausibility_gate_sensitivity.py`

Model diagnostics:

- `scripts/plots/plot_species_feature_importance.py`
- `scripts/plots/plot_species_partial_dependence.py`
- `scripts/plots/plot_species_use_observed_vs_predicted.py`

Seascape and component plots:

- `scripts/plots/plot_bayesian_gmm_component_maps.py`
- `scripts/plots/plot_bayesian_gmm_component_maps.py --monthly`
- `scripts/plots/plot_seascapes_maps.py`
- `scripts/plots/plot_seascape_species_use_monthly_matrix.py`
- `scripts/plots/plot_seascape_prediction_maps.py`
- `scripts/plots/plot_seascape_prediction_maps.py --monthly-matrix --matrix-values species_use_log_pred risk_log_pred`
- `scripts/plots/plot_prediction_latent_risk_monthly_matrix.py --model-name seascape_som_15x15_hierarchical_k30 --product-name joint --year 2022 --agg non_zero_mean --color-bin-source monthly_species --color-quantiles 0 0.55 0.80 0.95 1.0`

Gear-aware plots:

- `scripts/plots/plot_set_longline_bbal_risk_example.py`
- `scripts/plots/plot_weekly_gear_aware_risk_examples.py`

The visualization module is functional at this point, but several plot families
still carry presentation-specific scale and legend choices. Preserve those
settings until a deliberate visualization refactor has visual regression
references.

## Publishing Note

The exact public pipeline order is still being finalized. Use this document as
the current workflow map, and use `docs/publishing_checklist.md` for remaining
publishing decisions.
