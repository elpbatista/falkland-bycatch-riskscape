# Data Folder Assessment

This page is a release-time snapshot of how the local `data/` tree is used by
version `v0.1.0` of the Falkland Bycatch Riskscape workflow. It is kept as an
audit note, not as the primary restoration guide. To restore released data
products, use `docs/zenodo_manifest.md`.

`data/raw/` is intentionally excluded from the Zenodo bundle. Raw provider
files must be downloaded or restored through the provider-specific instructions
in `docs/datasets.md` and `docs/authentication.md`.

## Current Policy

- Keep code, notebooks, lightweight metadata, and selected static plot PNGs in
  Git.
- Keep generated data products out of Git.
- Restore reusable derived products from the Zenodo data bundle when the goal is
  to inspect notebooks, figures, and model outputs without rebuilding from raw
  inputs.
- Rebuild products from source data when testing the full workflow or adapting
  the template to a new system.

## Active Data Families

### `data/grids/`

Role:
exact H3 study frame and spatial grid products.

Used by:
feature construction, spatial joins, maps, diagnostics, and prediction products.

Release status:
included in `falkland-bycatch-riskscape-grids-v0.1.0.tar.gz`.

### `data/processed/`

Role:
lookup and index products, including H3 raster lookups, neighbor tables,
seasonal lookup products, and related preprocessing artifacts.

Used by:
feature builders, diagnostics, and reproducible joins between source products
and the H3 grid.

Release status:
included in `falkland-bycatch-riskscape-processed-v0.1.0.tar.gz`.

### `data/features/`

Role:
yearly environmental, fishing-effort, species-presence, and static feature
partitions.

Used by:
model table assembly, diagnostics, notebooks, and selected inspection products.

Release status:
included in `falkland-bycatch-riskscape-features-v0.1.0.tar.gz`.

Important caveat:
species-derived features are derived from telemetry support. They should be
interpreted as model-support products, not as complete population distributions
or unrestricted raw species data.

### `data/modeling/`

Role:
model-ready tables, environmental regimes, selected model artifacts,
plausibility products, prediction products, weekly operator products, and
curated validation metrics.

Used by:
model inspection, prediction notebooks, operational-output notebooks, plotting
scripts, and quality checks.

Release status:
included in `falkland-bycatch-riskscape-modeling-v0.1.0.tar.gz`.

Important subfolders:

- `feature_grid/`: model-facing daily H3 feature grid.
- `species_training/`: species-use training table.
- `fishing_training/`: retained model-facing fishing table; not currently the
  center of a selected fishing model.
- `models/extra_trees_som_hierarchical_k30_5fold_blockcv/`: selected
  species-use model artifact.
- `models/bayesian_gmm_k30/`: selected environmental plausibility model
  artifact.
- `environmental_regimes/`: active table for SOM-hierarchical k30 seascape
  assignments and Bayesian GMM k30 component assignments.
- `predictions/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/`:
  selected production prediction product.
- `plausibility/bayesian_gmm_k30/`: plausibility surfaces used to interpret
  environmental support.
- `weekly_operator/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/`:
  weekly latent-risk products and operational summaries.
- `seascape_species_use/som_15x15_hierarchical_k30/`: retained exploratory 2022
  seascape-conditioned species-use surface.
- `metrics/`: curated production metrics and release-facing validation
  summaries.

### `data/plot_exports/`

Role:
tabular products used by plotting scripts and diagnostics.

Used by:
operational maps, plausibility plots, species-use diagnostics, environmental
summaries, and seascape summaries.

Release status:
included in `falkland-bycatch-riskscape-plot_exports-v0.1.0.tar.gz`.

### `data/raw/`

Role:
local provider downloads and restricted source files.

Release status:
not included in the Zenodo data bundle and ignored by Git.

Restore path:
use `scripts/data/download_data.py` for provider-backed datasets where possible,
place approved local species data in the configured raw species location, and
use `docs/authentication.md` for credentials.

## Deprecated Products Removed Before Release

The release bundle excludes historical products that are no longer part of the
public workflow:

- KMeans seascape products and KMeans-based model outputs.
- Older SOM k15/k18 seascape variants.
- MBON seascape assignment products used only for the regional coverage
  assessment.
- Broad model sweeps and old exploratory model objects.
- Patch-only integrity checks and local system artifacts such as `.DS_Store`.

The selected public pathway is the SOM-hierarchical k30 environmental-regime
workflow with the Extra Trees species-use model and Bayesian GMM k30
plausibility layer.

## Regeneration Caveats

The public workflow is reusable and documented, but not every archived product
is currently guaranteed to regenerate from raw inputs through a single command.
The high-level pipeline can rebuild major stages, while some modeling,
seascape, plotting, and provider-download paths still need a future refactor for
full end-to-end automation.

Known caveats for future maintenance:

- GEBCO/CEDA bathymetry download handling still needs live-provider retesting.
- Visualization scripts are functional but should be refactored carefully with
  visual regression outputs available.
- The environmental-regime and model-training paths should be tightened before a
  future release claims complete raw-to-public-product regeneration.
