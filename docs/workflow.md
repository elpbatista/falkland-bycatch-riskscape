# Workflow

This document describes the intended public workflow at a high level. The
script inventory in `docs/script_inventory.md` tracks which entry points are
ready to keep and document.

## 1. Restore Reference Layers

Download public reference layers used by maps, spatial overlays, and study-area
setup:

```bash
python scripts/download_reference_data.py
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

Download configured datasets:

```bash
python scripts/download_data.py
```

To download selected datasets:

```bash
python scripts/download_data.py --dataset sst chl ssh wind gfw
```

Raw downloads are written under `data/raw/` and are ignored by Git.

## 4. Build Spatial and Feature Tables

The workflow builds a common H3 spatial frame, then aligns environmental,
fishing-effort, and species-presence features to that frame.

Core script candidates include:

- `scripts/build_grid.py`
- `scripts/build_static_features.py`
- `scripts/build_environmental_feature_table.py`
- `scripts/build_fishing_effort_feature_table.py`
- `scripts/build_species_presence_feature_table.py`
- `scripts/build_model_datasets.py`

## 5. Train, Predict, and Evaluate

Modeling scripts currently kept for public workflow documentation are:

- `scripts/train_models.py`
- `scripts/predict_models.py`
- `scripts/evaluate_models.py`

## 6. Plot Outputs

Generated plots are written under `plots/` and ignored by Git. Selected figures
or reproducibility bundles should be archived externally on Zenodo.

## Publishing Note

The exact public pipeline order is still being finalized. Use this document as
the current workflow map, and use `docs/publishing_checklist.md` for remaining
publishing decisions.
