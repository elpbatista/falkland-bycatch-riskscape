# Script Inventory

This working inventory tracks scripts that are candidates for the reusable
riskscape workflow documentation.

Status values:

- `keep`: include in the documented workflow.
- `review`: likely useful, but needs a pass before documenting.
- `defer`: specialized, exploratory, or report-specific; keep out of the main
  workflow until deliberately promoted.

## Reference and Data Acquisition

- `keep` `scripts/run_pipeline.py`: Runs high-level workflow stages through
  `src/riskscape/workflow.py`. This is the preferred public entry point once
  source and reference data are available.
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
- `defer` `scripts/build_mbon_seascape_assignments.py`: Builds NOAA/MBON
  8-day seascape assignments on the project H3/date grid.
- `defer` `scripts/classify_environmental_seascapes.py`: Classifies
  environmental seascapes and compares them with GMM components.
- `defer` `scripts/classify_som_hierarchical_seascapes.py`: Classifies
  environmental seascapes with SOM prototypes and clustering.
- `review` `scripts/build_seascape_species_use_surfaces.py`: Builds species-use
  surfaces from seascape assignments.
- `defer` `scripts/assign_components.py`: Assigns dominant Bayesian/GMM
  ecological components.
- `defer` `scripts/build_weekly_operator_latent_risk.py`: Builds weekly
  latent-risk products for operator-facing maps.
- `defer` `scripts/export_fishing_effort_by_gear_flag.py`: Exports raw GFW
  fishing effort by H3/date/gear/flag for examples.
- `defer` `scripts/fix_date_utc.py`: One-off date dtype repair utility.
- `defer` `scripts/remove_columns.py`: One-off column removal utility.

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
- `defer` `scripts/analyze_seascape_validation_designs.py`: Quantifies
  seascape class balance and species-training support.
- `defer` `scripts/compare_bayesian_gmm_components.py`: Compares joint
  Bayesian GMM species models with different component counts.
- `defer` `scripts/inspect_feature_importance.py`: Inspects model feature
  importance.
- `defer` `scripts/inspect_gmm_bayesian.py`: Inspects joint Bayesian GMM
  ecological regimes.
- `defer` `scripts/run_block_cv_variant_comparison.py`: Runs BlockCV validation
  variants and consolidates report-ready metrics.
- `defer` `scripts/summarize_bayesian_gmm_component_tables.py`: Summarizes
  Bayesian/GMM environmental components for report tables.
- `defer` `scripts/summarize_kmeans_seascape_species_tables.py`: Summarizes
  observed species-use records by KMeans seascape class.

## Modeling

- `keep` `scripts/build_model_datasets.py`: Assembles model-ready datasets.
- `keep` `scripts/train_models.py`: Trains riskscape models.
- `keep` `scripts/predict_models.py`: Generates model predictions.
- `keep` `scripts/evaluate_models.py`: Evaluates trained models and predictions.
- `defer` `scripts/train_active_species_model.py`: Trains the active production
  species-use model; needs relationship review with `train_models.py`.

## Visualization and Diagnostics

- `keep` `scripts/plot_study_area_map.py`: Plots the study area and reference
  overlay layers.
- `review` `scripts/plot_bathymetry_map.py`: Plots H3 bathymetry with land
  overlay.
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
- `review` `scripts/plot_prediction_latent_risk_monthly_matrix.py`: Plots
  monthly latent-risk matrices from prediction outputs.
- `review` `scripts/plot_species_feature_importance.py`: Plots feature
  importance for selected species-use models.
- `review` `scripts/plot_species_partial_dependence.py`: Plots manual partial
  dependence for selected species-use models.
- `review` `scripts/plot_species_use_observed_vs_predicted.py`: Plots observed
  versus predicted species-use values.
- `review` `scripts/plot_plausibility_maps.py`: Plots Bayesian GMM
  environmental plausibility maps.
- `review` `scripts/plot_plausibility_monthly_climatology.py`: Plots monthly
  environmental plausibility climatologies.
- `review` `scripts/plot_plausibility_yearly_timeseries.py`: Plots yearly
  environmental plausibility summaries.
- `review` `scripts/plot_plausibility_gate_sensitivity.py`: Plots latent-risk
  sensitivity to plausibility-gate strength.
- `review` `scripts/plot_bayesian_gmm_component_maps.py`: Plots dominant
  Bayesian/GMM environmental component assignments.
- `review` `scripts/plot_seascapes_maps.py`: Plots feature-only KMeans seascape
  assignments.
- `review` `scripts/plot_seascape_species_use_monthly_matrix.py`: Plots monthly
  seascape-conditioned species-use maps.
- `review` `scripts/plot_seascape_prediction_maps.py`: Plots seascape
  prediction maps; needs naming/relationship review with
  `plot_prediction_maps.py`.
- `defer` `scripts/plot_set_longline_bbal_risk_example.py`: Plots one
  gear-aware realized-risk example for BBAL set longlines.
- `defer` `scripts/plot_weekly_gear_aware_risk_examples.py`: Plots weekly
  gear-aware realized-risk examples.
- `defer` `scripts/plot_weekly_latent_risk_with_jigger_activity.py`: Plots
  weekly latent risk with fishing-activity cells marked.
- `defer` `scripts/plot_weekly_operator_fisheries_grid_example.py`: Plots weekly
  latent-risk climatology aggregated to fisheries grid squares.
- `defer` `scripts/plot_weekly_operator_latent_risk.py`: Plots weekly
  latent-risk operator climatology maps.
- `review` `scripts/plot_all_maps.py`: Convenience plotting entry point; needs
  review before documenting as public workflow.
- `defer` `scripts/test_prediction_maps.py`: Legacy prediction-map plotting
  test; review before keeping as a public script.
