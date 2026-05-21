# Falkland Bycatch Riskscape Workflow

Reusable workflow components for building dynamic bycatch riskscapes from
species observations, fishing effort, environmental data, and spatial reference
layers.

The repository is organized as a Python package plus executable workflow
scripts. The Falkland Islands case study is the worked example, but the project
is being prepared as a template that can be adapted to other regions, species,
and fisheries.

## What This Repository Contains

- `src/riskscape/`: reusable Python package code.
- `scripts/`: workflow entry points for downloading, feature building, modeling,
  prediction, evaluation, and plotting.
- `notebooks/`: tutorial, presentation, and inspection notebooks for selected
  workflow stages.
- `docs/`: current workflow, data, authentication, publishing, and script
  inventory notes.
- `config.yaml`: canonical Falkland Islands workflow configuration.
- `reference/README.md`: instructions for restoring public reference layers.

Generated data, plots, downloaded reference layers, and model outputs are not
stored in Git.

## Data Policy

This repository tracks code, documentation, and lightweight configuration. It
does not track generated or downloaded data folders such as:

- `data/`
- `plots/`
- `outputs/`
- downloaded files under `reference/`

Reference layers can be restored with:

```bash
python scripts/download_reference_data.py
```

Larger release bundles for derived data, selected plots, and model outputs are
intended to be archived externally on Zenodo.

Some source datasets may require provider credentials, access approval, or
collaborator permission. See `docs/datasets.md` and `docs/authentication.md`.

## Setup

Create and activate a Python environment, then install the package in editable
mode:

```bash
pip install -e .
```

To install optional data-provider clients used by download scripts:

```bash
pip install -e ".[downloads]"
```

To install optional SOM/seascape exploration dependencies:

```bash
pip install -e ".[seascapes]"
```

Copy the environment template if you need local credentials:

```bash
cp .env.example .env
```

Edit `.env` locally. Do not commit it.

`config.yaml` is the canonical public workflow configuration. Local secrets and
runtime overrides belong in `.env`. Notebooks are not the pipeline orchestrator;
they are used to present, demonstrate, and inspect workflow stages.

## Basic Workflow

The public workflow is still being formalized. The current high-level sequence
is:

1. Restore public reference layers.
2. Download configured environmental and fishing-effort data.
3. Build the spatial grid and lookup tables.
4. Build environmental, fishing-effort, and species-presence feature tables.
5. Assemble model-ready datasets.
6. Train and evaluate models.
7. Generate predictions and plots.

See `docs/workflow.md` for the current workflow map. The script inventory in
`docs/script_inventory.md` tracks which scripts are intended to remain part of
the documented workflow.

After reference layers and source data are available, the grouped workflow
entry point is:

```bash
python scripts/run_pipeline.py --stage all
```

Use `--stage all-with-downloads` to include external downloads, or run a named
stage such as `spatial`, `features`, `model-tables`, `modeling`, or `checks`.

## Credentials

Credentials must live outside committed files. The current convention is:

- Earthdata credentials in `~/.netrc`
- Copernicus Marine credentials through `copernicusmarine login`
- CDS credentials in `~/.cdsapirc`
- Global Fishing Watch token in local `.env` as `GFW_TOKEN`

See `docs/authentication.md` for details.

## Publishing Status

This repository is mid-refactor for public release. The current checklist is in
`docs/publishing_checklist.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. A Zenodo DOI will be added
when the first public release/archive is created.
