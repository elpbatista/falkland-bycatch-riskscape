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
- `scripts/`: grouped workflow entry points; `scripts/run_pipeline.py` is the
  preferred public front door.
- `notebooks/`: tutorial, presentation, and inspection notebooks for selected
  workflow stages.
- `docs/`: current workflow, data, authentication, notebook, publishing, and
  script inventory notes.
- `config.yaml`: canonical Falkland Islands workflow configuration.
- `reference/README.md`: instructions for restoring public reference layers.

Generated data, downloaded reference layers, and model outputs are not
stored in Git. Selected static plot PNGs are tracked because the public
notebooks use them as explanatory figures.

## Data Policy

This repository tracks code, documentation, and lightweight configuration. It
does not track generated or downloaded data folders such as:

- `data/`
- `outputs/`
- downloaded files under `reference/`

Selected static PNG figures under `plots/` are tracked for the public
notebooks. Large animations, intermediate frames, and tabular plot exports stay
outside Git and are included in the Zenodo data bundle when needed.

Reference layers can be restored with:

```bash
python scripts/data/download_reference_data.py
```

Larger release bundles for derived data, selected plots, and model outputs are
archived externally on Zenodo. The data-bundle concept DOI, which resolves to
the latest version, is:

<https://doi.org/10.5281/zenodo.20334806>

Zenodo recommends citing the current data bundle as:

Batista Echevarría, J. L. (2026). Falkland Bycatch Riskscape Data Bundle [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.20337229>

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
the documented workflow, including plot commands with custom scale settings.
Notebook roles and cleanup needs are tracked in `docs/notebooks.md`.

After reference layers and source data are available, the grouped workflow
entry point is:

```bash
python scripts/run_pipeline.py --stage all
```

Use `--stage all-with-downloads` to include external downloads, or run a named
stage such as `spatial`, `features`, `model-tables`, `modeling`, or `checks`.

Plotting is grouped separately:

```bash
python scripts/plots/plot_all_maps.py --group context
python scripts/plots/plot_all_maps.py --group predictions
python scripts/plots/plot_all_maps.py --group seascapes
python scripts/plots/plot_all_maps.py --group weekly
```

Use `--list` before running a group to inspect the exact commands.

## Credentials

Credentials must live outside committed files. The current convention is:

- Earthdata credentials in `~/.netrc`
- Copernicus Marine credentials through `copernicusmarine login`
- CDS credentials in `~/.cdsapirc`
- Global Fishing Watch token in local `.env` as `GFW_TOKEN`

See `docs/authentication.md` for details.

## Publishing Status

This repository is mid-refactor for public release. The current checklist is in
`docs/publishing_checklist.md`. The visualization scripts are functional but
will need a careful second-pass refactor after the release-critical cleanup.

## Citation

Citation metadata are provided in `CITATION.cff`. The repository/software DOI is:

<https://doi.org/10.5281/zenodo.20348906>

The associated Zenodo data bundle has a concept DOI for the latest version:

<https://doi.org/10.5281/zenodo.20334806>

Zenodo recommends citing the current data bundle as:

Batista Echevarría, J. L. (2026). Falkland Bycatch Riskscape Data Bundle [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.20337229>
