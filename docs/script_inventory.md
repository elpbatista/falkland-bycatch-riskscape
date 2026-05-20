# Script Inventory

This working inventory tracks scripts that are candidates for the reusable
riskscape workflow documentation.

Status values:

- `keep`: include in the documented workflow.
- `review`: likely useful, but needs a pass before documenting.

## Reference and Data Acquisition

- `keep` `scripts/download_reference_data.py`: Downloads public reference
  layers used by maps and overlays, including Natural Earth and SAERI-hosted
  Falklands layers where direct portal resources are available.
- `keep` `scripts/download_data.py`: Downloads configured model input datasets
  through provider modules.

## Spatial Framework, Lookups, and Indices

- `keep` `scripts/build_grid.py`: Builds the project spatial grid.
- `keep` `scripts/build_static_features.py`: Builds static grid-level features.
- `keep` `scripts/build_h3_lookup.py`: Builds lookup tables between source
  rasters/products and H3 cells.
- `keep` `scripts/build_neighbor_table.py`: Builds grid-cell neighbor
  relationships.
- `keep` `scripts/build_neighbor_index_table.py`: Builds indexed neighbor
  relationships for downstream features.
- `keep` `scripts/build_seasonal_lookup.py`: Builds seasonal lookup tables used
  by temporal summaries.

## Feature Construction

- `keep` `scripts/build_environmental_feature_table.py`: Builds environmental
  feature tables from raw/downloaded products.
- `keep` `scripts/build_fishing_effort_feature_table.py`: Builds fishing-effort
  features.
- `keep` `scripts/build_species_presence_feature_table.py`: Builds
  species-presence features.
- `keep` `scripts/build_derived_features.py`: Builds derived model-facing
  features from primary feature tables.
- `review` `scripts/build_environmental_anomalies.py`: Builds environmental
  anomaly features.
- `review` `scripts/build_environmental_gradients.py`: Builds environmental
  gradient features.
- `review` `scripts/build_environmental_regime_table.py`: Builds environmental
  regime/seascape tables.
- `review` `scripts/build_seascape_species_use_surfaces.py`: Builds species-use
  surfaces from seascape assignments.

## Inspection and Validation

- `keep` `scripts/feature_qa_summary.py`: Summarizes feature completeness and
  quality checks.
- `keep` `scripts/inspect_columns.py`: Inspects table schemas during pipeline
  checks.
- `keep` `scripts/quick_validation.py`: Runs lightweight validation checks over
  generated products.
- `review` `scripts/run_correlation.py`: Runs correlation analysis for feature
  inspection.
- `review` `scripts/run_relationships.py`: Runs relationship diagnostics for
  feature/model inspection.

## Modeling

- `keep` `scripts/build_model_datasets.py`: Assembles model-ready datasets.
- `keep` `scripts/train_models.py`: Trains riskscape models.
- `keep` `scripts/predict_models.py`: Generates model predictions.
- `keep` `scripts/evaluate_models.py`: Evaluates trained models and predictions.

## Visualization and Diagnostics

- `keep` `scripts/plot_study_area_map.py`: Plots the study area and reference
  overlay layers.
- `keep` `scripts/plot_environmental_histograms.py`: Plots environmental
  feature distributions for inspection.
- `keep` `scripts/plot_environmental_correlation_heatmap.py`: Plots
  environmental feature correlations.
- `review` `scripts/plot_environmental_daily_timeseries.py`: Plots daily
  environmental time series diagnostics.
- `review` `scripts/plot_environmental_monthly_matrix.py`: Plots monthly
  environmental matrices.
- `review` `scripts/plot_environmental_gradient_maps.py`: Plots environmental
  gradient products.
- `review` `scripts/plot_environmental_single_date_maps.py`: Plots single-date
  environmental maps.
- `keep` `scripts/plot_fishing_activity_map.py`: Plots fishing activity maps for
  inspection.
- `review` `scripts/plot_fishing_activity_monthly_matrix.py`: Plots monthly
  fishing-activity matrices.
- `review` `scripts/plot_fishing_activity_monthly_timeseries.py`: Plots monthly
  fishing-activity time series.
- `keep` `scripts/plot_species_presence_maps.py`: Plots species-presence maps
  for inspection.
- `review` `scripts/plot_relationship_diagnostics.py`: Plots relationship
  diagnostics.
- `keep` `scripts/plot_prediction_maps.py`: Plots model prediction maps.
- `review` `scripts/plot_seascape_prediction_maps.py`: Plots seascape
  prediction maps; needs naming/relationship review with
  `plot_prediction_maps.py`.
- `review` `scripts/plot_all_maps.py`: Convenience plotting entry point; needs
  review before documenting as public workflow.
