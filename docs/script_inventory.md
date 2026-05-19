# Script Inventory

This is a working inventory for scripts that should be kept and documented as
part of the reusable riskscape workflow. It is intentionally conservative: a
script appears here when it has a clear role in the public workflow or in
rebuilding reference/project assets.

Status values:

- `keep`: include in the documented workflow.
- `review`: likely useful, but needs a pass before documenting.
- `dev`: development-only or exploratory; not part of the public workflow yet.

## Reference and Data Acquisition

| Script                               | Status | Role                                                                                                                                                                |
|--------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scripts/download_reference_data.py` | keep   | Downloads public reference layers used by maps and overlays, including Natural Earth and SAERI-hosted Falklands layers where direct portal resources are available. |
| `scripts/download_data.py`           | keep   | Downloads configured model input datasets through provider modules.                                                                                                 |

## Core Workflow Candidates

| Script                                            | Status | Role                                                              |
|---------------------------------------------------|--------|-------------------------------------------------------------------|
| `scripts/build_grid.py`                           | keep   | Builds the project spatial grid.                                  |
| `scripts/build_static_features.py`                | keep   | Builds static grid-level features.                                |
| `scripts/build_environmental_feature_table.py`    | keep   | Builds environmental feature tables from raw/downloaded products. |
| `scripts/build_environmental_anomalies.py`        | review | Builds environmental anomaly features.                            |
| `scripts/build_environmental_gradients.py`        | review | Builds environmental gradient features.                           |
| `scripts/build_fishing_effort_feature_table.py`   | keep   | Builds fishing-effort features.                                   |
| `scripts/build_species_presence_feature_table.py` | keep   | Builds species-presence features.                                 |
| `scripts/build_model_datasets.py`                 | keep   | Assembles model-ready datasets.                                   |
| `scripts/train_models.py`                         | keep   | Trains riskscape models.                                          |
| `scripts/predict_models.py`                       | keep   | Generates model predictions.                                      |
| `scripts/evaluate_models.py`                      | keep   | Evaluates trained models and predictions.                         |

## Visualization Candidates

| Script                                     | Status | Role                                                                                             |
|--------------------------------------------|--------|--------------------------------------------------------------------------------------------------|
| `scripts/plot_study_area_map.py`           | keep   | Plots the study area and reference overlay layers.                                               |
| `scripts/plot_prediction_maps.py`          | keep   | Plots model prediction maps.                                                                     |
| `scripts/plot_seascape_prediction_maps.py` | review | Plots seascape prediction maps; needs naming/relationship review with `plot_prediction_maps.py`. |
| `scripts/plot_all_maps.py`                 | review | Convenience plotting entry point; needs review before documenting as public workflow.            |
