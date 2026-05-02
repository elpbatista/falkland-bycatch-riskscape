# Workflow

1. Ingest raw datasets for environmental conditions, species presence, fishing activity, and bycatch events.

2. Clean and filter datasets to remove invalid or missing records.

3. Aggregate all data to the H3 grid at resolution 6.

4. Align all datasets to a common daily temporal resolution.

5. Compute derived variables, including gradients and anomalies.

6. Standardize variables for consistent scaling.

7. Validate the resulting dataset using range, consistency, and statistical checks.

8. Assemble the final dataset at the cell × day level.

9. Apply the modeling framework to estimate hazard and bycatch risk.

10. Generate outputs for analysis and visualization.

## Structure

```text
bycatch-riskscape/
├── config.yaml
├── pyproject.toml
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── grids/
│
├── docs/
│   ├── model_spec.md
│   ├── workflow.md
│   └── figures/
│
├── logs/
│
├── notebooks/
│   ├── 00_reproduce_pipeline.ipynb
│   ├── 01_data_sources_and_scope.ipynb
│   ├── 02_h3_grid_checks.ipynb
│   ├── 03_temporal_representation.ipynb
│   ├── 04_gradients_and_anomalies.ipynb
│   ├── 05_dataset_assembly.ipynb
│   ├── 06_model_structure.ipynb
│   └── 07_outputs_and_validation.ipynb
│
├── plots/
│
├── scripts/
│   ├── build_grid.py
│   ├── download_environmental.py
│   ├── build_layer1.py
│   ├── assemble_dataset.py
│   └── validate_dataset.py
│
└── src/
    └── riskscape/
        ├── __init__.py
        ├── config.py
        │
        ├── providers/
        │   ├── __init__.py
        │   ├── cds.py
        │   ├── copernicus.py
        │   └── podaac.py
        │
        ├── grid/
        │   ├── __init__.py
        │   └── h3_grid.py
        │
        ├── processing/
        │   ├── __init__.py
        │   ├── aggregate.py
        │   ├── align.py
        │   ├── anomalies.py
        │   ├── gradients.py
        │   └── standardize.py
        │
        ├── datasets/
        │   ├── __init__.py
        │   ├── io.py
        │   ├── schema.py
        │   └── warehouse.py
        │
        ├── modeling/
        │   ├── __init__.py
        │   ├── species.py
        │   ├── hazard.py
        │   └── risk.py
        │
        └── validation/
            ├── __init__.py
            ├── checks.py
            └── summaries.py
```
