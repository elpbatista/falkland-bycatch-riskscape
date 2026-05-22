# Script Inventory

This inventory documents the current public script surface for the Falkland
Bycatch Riskscape workflow. It is a maintenance reference, not a substitute for
`docs/workflow.md` or the public notebooks.

Status values:

- `public`: intended for normal documented use.
- `support`: useful inspection, diagnostic, or specialized script; public, but
  not a primary workflow entry point.
- `experimental`: retained for exploratory or legacy analyses; review before
  using as a template for new work.
- `private`: intentionally outside the public workflow.

## Top-Level Entry Point

- `public` `scripts/run_pipeline.py`: grouped pipeline entry point. It delegates
  workflow stages to `src/riskscape/workflow.py` and is the preferred public
  command once reference layers and source data are available.

## Data Acquisition

- `public` `scripts/data/download_reference_data.py`: restores public reference
  layers used by maps and overlays, including Natural Earth and SAERI-hosted
  Falklands layers where direct resources are available.
- `public` `scripts/data/download_data.py`: downloads configured provider-backed
  input datasets through provider modules. Some providers require credentials;
  see `docs/authentication.md`.

## Spatial Framework, Lookups, and Indices

- `public` `scripts/build/build_grid.py`: builds the configured H3 study grid.
- `public` `scripts/build/build_static_features.py`: builds static grid-level
  features.
- `public` `scripts/build/build_h3_lookup.py`: builds source-product to H3 lookup
  tables.
- `public` `scripts/build/build_neighbor_table.py`: builds H3 neighbor
  relationships.
- `public` `scripts/build/build_neighbor_index_table.py`: builds indexed neighbor
  relationships used by downstream feature calculations.
- `public` `scripts/build/build_seasonal_lookup.py`: builds seasonal lookup
  tables used by temporal summaries and anomaly logic.

## Feature Construction

- `public` `scripts/build/build_environmental_feature_table.py`: builds yearly
  environmental feature tables from downloaded environmental products.
- `public` `scripts/build/build_fishing_effort_feature_table.py`: builds fishing
  effort features on the H3/day grid.
- `public` `scripts/build/build_species_presence_feature_table.py`: builds
  species-presence support features from approved local species data.
- `public` `scripts/build/build_derived_features.py`: builds derived model-facing
  features from primary feature tables.
- `support` `scripts/build/build_environmental_anomalies.py`: builds or rebuilds
  environmental anomaly fields.
- `support` `scripts/build/build_environmental_gradients.py`: builds or rebuilds
  environmental gradient fields.
- `support` `scripts/build/build_environmental_regime_table.py`: builds the
  consolidated environmental-regime table used by selected SOM-hierarchical
  seascapes and Bayesian GMM components.
- `support` `scripts/build/build_model_datasets.py`: assembles model-ready
  species and fishing tables.
- `support` `scripts/build/build_weekly_operator_latent_risk.py`: builds weekly
  latent-risk products for operational maps and summaries.
- `support` `scripts/build/build_seascape_species_use_surfaces.py`: builds the
  seascape-conditioned species-use surfaces used by exploratory 2022 seascape
  matrices.
- `experimental` `scripts/build/build_mbon_seascape_assignments.py`: builds the
  NOAA/MBON 8-day seascape assignment diagnostic. Retained for comparison, not
  part of the selected production pathway.

## Modeling

- `public` `scripts/model/train_models.py`: trains baseline model candidates and
  diagnostics.
- `public` `scripts/model/predict_models.py`: generates model predictions.
- `public` `scripts/model/evaluate_models.py`: evaluates trained models and
  prediction products.
- `support` `scripts/model/train_active_species_model.py`: trains the active
  production species-use model used by the public release.
- `experimental` `scripts/model/run_block_cv_variant_comparison.py`: runs broader
  BlockCV validation variants. Retained for method development and comparison.

## Quality Assurance and Inspection

- `public` `scripts/qa/feature_qa_summary.py`: summarizes feature completeness,
  duplicate keys, missing values, and yearly coverage.
- `public` `scripts/qa/inspect_columns.py`: inspects table schemas during checks.
- `public` `scripts/qa/quick_validation.py`: runs lightweight validation over key
  generated products.
- `support` `scripts/qa/run_correlation.py`: runs environmental correlation
  analysis for feature inspection.
- `support` `scripts/qa/run_relationships.py`: runs relationship diagnostics for
  feature/model inspection.
- `support` `scripts/qa/inspect_feature_importance.py`: inspects selected model
  feature importance.
- `support` `scripts/qa/inspect_gmm_bayesian.py`: inspects Bayesian/GMM regime
  outputs.
- `support` `scripts/qa/analyze_seascape_validation_designs.py`: summarizes
  seascape class balance and species-training support for validation design.
- `support` `scripts/qa/compare_bayesian_gmm_components.py`: compares Bayesian
  GMM component-count options.
- `support` `scripts/qa/summarize_bayesian_gmm_component_tables.py`: summarizes
  Bayesian GMM environmental components.
- `experimental` `scripts/qa/summarize_kmeans_seascape_species_tables.py`:
  historical KMeans seascape summary utility retained for provenance only.

## Plotting and Visualization

The grouped plotting entry point is:

- `public` `scripts/plots/plot_all_maps.py`: runs plot groups such as `context`,
  `environmental`, `predictions`, `seascapes`, `weekly`, `gear`, `videos`, and
  `all`. Use `--list` to inspect the exact commands before running.

Context and source-product plots:

- `public` `scripts/plots/plot_study_area_map.py`: plots study area and reference
  overlays.
- `support` `scripts/plots/plot_bathymetry_map.py`: standalone bathymetry map;
  useful for inspection, but bathymetry is usually used through shared map
  context.
- `support` `scripts/plots/plot_environmental_histograms.py`: plots environmental
  feature distributions.
- `support` `scripts/plots/plot_environmental_correlation_heatmap.py`: plots
  environmental feature correlations.
- `support` `scripts/plots/plot_environmental_daily_timeseries.py`: plots daily
  environmental time series diagnostics.
- `support` `scripts/plots/plot_environmental_monthly_matrix.py`: plots monthly
  environmental matrices.
- `support` `scripts/plots/plot_environmental_gradient_maps.py`: plots
  environmental gradient products.
- `support` `scripts/plots/plot_environmental_single_date_maps.py`: plots
  single-date environmental maps.
- `support` `scripts/plots/plot_fishing_activity_map.py`: plots fishing activity
  maps.
- `support` `scripts/plots/plot_fishing_activity_monthly_matrix.py`: plots
  monthly fishing-activity matrices.
- `support` `scripts/plots/plot_fishing_activity_monthly_timeseries.py`: plots
  monthly fishing-activity time series.
- `support` `scripts/plots/plot_species_presence_maps.py`: plots species-presence
  support maps.
- `support` `scripts/plots/plot_relationship_diagnostics.py`: plots relationship
  diagnostics.

Prediction, plausibility, and diagnostics plots:

- `public` `scripts/plots/plot_prediction_maps.py`: plots selected production
  prediction maps.
- `public` `scripts/plots/plot_prediction_latent_risk_monthly_matrix.py`: plots
  monthly latent-risk matrices. The accepted seascape latent-risk scale uses:
  `--model-name seascape_som_15x15_hierarchical_k30 --product-name joint --year
  2022 --agg non_zero_mean --color-bin-source monthly_species --color-quantiles
  0 0.55 0.80 0.95 1.0`.
- `support` `scripts/plots/plot_species_feature_importance.py`: plots feature
  importance for selected species-use models.
- `support` `scripts/plots/plot_species_partial_dependence.py`: plots manual
  partial dependence for selected species-use models.
- `support` `scripts/plots/plot_species_use_observed_vs_predicted.py`: plots
  observed versus predicted species-use values.
- `support` `scripts/plots/plot_plausibility_maps.py`: plots Bayesian GMM
  environmental plausibility maps.
- `support` `scripts/plots/plot_plausibility_monthly_climatology.py`: plots
  monthly environmental plausibility climatologies.
- `support` `scripts/plots/plot_plausibility_yearly_timeseries.py`: plots yearly
  environmental plausibility summaries.
- `support` `scripts/plots/plot_plausibility_gate_sensitivity.py`: plots
  latent-risk sensitivity to plausibility-gate strength.
- `support` `scripts/plots/plot_bayesian_gmm_component_maps.py`: plots dominant
  Bayesian GMM environmental component assignments.
- `support` `scripts/plots/plot_seascapes_maps.py`: plots selected seascape
  assignments.
- `support` `scripts/plots/plot_seascape_species_use_monthly_matrix.py`: plots
  seascape-conditioned species-use matrices.
- `support` `scripts/plots/plot_seascape_prediction_maps.py`: plots
  seascape-conditioned prediction maps and matrices.

Operational plots:

- `support` `scripts/plots/plot_weekly_operator_latent_risk.py`: plots weekly
  latent-risk operator climatology maps and animation-oriented products.
- `support` `scripts/plots/plot_weekly_operator_fisheries_grid_example.py`:
  plots weekly latent-risk climatology aggregated to fisheries grid squares.
- `support` `scripts/plots/plot_weekly_latent_risk_with_jigger_activity.py`:
  plots weekly latent risk with fishing-activity cells marked, including flag
  filters.
- `support` `scripts/plots/plot_weekly_gear_aware_risk_examples.py`: plots weekly
  gear-aware realized-risk examples.
- `support` `scripts/plots/plot_set_longline_bbal_risk_example.py`: plots a
  specialized BBAL set-longline example.
- `experimental` `scripts/plots/test_prediction_maps.py`: legacy prediction-map
  plotting test retained for comparison only.

## Tools and Compatibility Utilities

- `support` `scripts/tools/classify_environmental_seascapes.py`: environmental
  seascape classification and comparison utility.
- `support` `scripts/tools/classify_som_hierarchical_seascapes.py`: selected SOM
  prototype and hierarchical clustering workflow for environmental seascapes.
- `support` `scripts/tools/assign_components.py`: assigns dominant Bayesian GMM
  components.
- `support` `scripts/tools/export_fishing_effort_by_gear_flag.py`: exports H3/day
  fishing effort by gear and flag for operational examples.
- `experimental` `scripts/tools/fix_date_utc.py`: one-off date dtype repair
  utility.
- `experimental` `scripts/tools/remove_columns.py`: one-off column removal
  utility.
- `support` `scripts/riskscape/__init__.py`: import shim retained for scripts
  while the public command structure settles. It can likely be removed in a
  future refactor once all public scripts rely on `pip install -e .` or a shared
  bootstrap pattern.

## Private Development Scripts

`scripts/dev/` is intentionally excluded from Git and from the public workflow.
It contains local exploratory helpers and should not be documented as part of
normal use.
