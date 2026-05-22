# Zenodo Data Manifest

This page records the current decision about which local products are useful
enough to upload to Zenodo alongside a repository release.

Zenodo concept DOI, latest version:
<https://doi.org/10.5281/zenodo.20334806>

Recommended citation DOI:
<https://doi.org/10.5281/zenodo.20337229>

It is a working manifest, not a final upload receipt. Update it after the
archive is finalized with the version and exact file names.

This manifest is based on the non-raw data assessment in
`docs/data_folder_assessment.md`. Raw provider downloads under `data/raw/` are
not assessed here and are not Zenodo candidates.

## Recommended Upload

These products are the strongest candidates because they are reusable results
from the workflow and are not simply raw provider downloads.

### Production Predictions

Path:
`data/modeling/predictions/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/`

Size:
part of the `2.9G` predictions folder.

Reason:
main annual gridded prediction products for the selected production model.

### Weekly Operational Products

Path:
`data/modeling/weekly_operator/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/`

Size:
part of the `444M` weekly-operator folder.

Reason:
operational weekly latent-risk products, including climatology and 2022
sequence outputs.

### Weekly Plot Exports

Path:
`data/plot_exports/weekly_operator/hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30/`

Size:
part of the `2.1M` weekly plot exports folder.

Reason:
small tabular summaries that support the operational figures.

### H3 Study Grid

Path:
`data/grids/`

Size:
`36M`.

Reason:
exact H3 study frame used by the workflow. Including it avoids ambiguity about
the spatial domain and lets users inspect products without rebuilding the grid.

### Processed Lookups and Indices

Path:
`data/processed/`

Size:
`151M`.

Reason:
reusable lookup and index products, including H3 lookups, neighbor tables,
seasonal lookup, and static/environmental lookup outputs.

### Environmental Regimes

Path:
`data/modeling/environmental_regimes/`

Size:
`674M`.

Reason:
consolidated H3/date environmental-regime table. It stores the selected SOM
hierarchical k30 seascape assignment (`som_prototype`, `seascape`,
`seascape_distance`) together with Bayesian GMM k30 component assignments
(`bayesian_gmm_k30_component`, probability, and entropy). These fields support
seascape maps, component maps, blocked diagnostics, and seascape-conditioned
products.

### Seascape-Conditioned Species Use

Path:
`data/modeling/seascape_species_use/som_15x15_hierarchical_k30/`

Size:
`156M`.

Reason:
experimental 2022 species-use surface conditioned on the selected SOM
seascapes. It supports the 2022 seascape-conditioned species-use and risk
matrices and can be regenerated with
`python3 scripts/build/build_seascape_species_use_surfaces.py --years 2022`.

### Plausibility Products

Path:
`data/modeling/plausibility/bayesian_gmm_k30/`

Size:
part of the `1.1G` plausibility folder.

Reason:
plausibility surfaces used to combine environmental support with species-use
predictions.

### Metrics

Paths:

- `data/modeling/metrics/species_model_extra_trees_som_hierarchical_k30_5fold_blockcv_production_metrics.csv`
- `data/modeling/metrics/species_model_block_cv_selected_som_k30_5fold.csv`
- `data/modeling/metrics/species_model_block_cv_selected_som_k30_5fold.md`
- `data/modeling/metrics/bayesian_gmm_component_comparison.csv`
- `data/modeling/metrics/seascapes/seascape_summary_som_15x15_hierarchical_k30_2014-2023.csv`

Size:
small.

Reason:
production model metrics, selected model block-CV summaries, Bayesian GMM
component comparison used for plausibility decisions, and the selected
seascape summary.

### Final Figures

Path:
`plots/`

Size:
`580M`.

Reason:
final figure bundle for inspection, presentation, and reuse.

## Do Not Upload

These products should not go to Zenodo as part of the reusable project archive.

### Deprecated Model Outputs

Scope:
deprecated KMeans products, older SOM k15/k18 variants, MBON seascape products,
broad model sweeps, and older exploratory model objects were removed locally
during publishing cleanup.

Reason:
the public workflow now centers on the SOM hierarchical k30 pathway and the
selected hybrid Extra Trees plus Bayesian GMM k30 production model.

### Local System Artifacts

Scope:
`.DS_Store` files.

Reason:
these are local system artifacts.

## Conditional Upload

These may be useful, but need a deliberate decision before inclusion.

### Feature Tables

Path:
`data/features/`

Decision needed:
large feature tables; include only if the goal is to let users skip feature
construction.

### Model Objects

Path:
`data/modeling/models/`

Decision needed:
model objects can be useful, but only if they are stable, documented, and
loadable across environments.

### Plot Exports

Path:
`data/plot_exports/`

Decision needed:
include selected subfolders when they support published figures or operational
outputs.

## Suggested Zenodo Package Layout

```text
zenodo/
  README.md
  predictions/
  grids/
  processed/
  weekly_operator/
  environmental_regimes/
  plausibility/
  metrics/
  plot_exports/
  plots/
```

The archive README should state the repository release tag used to generate the
files, the date of generation, the selected model name, and any data that are
not redistributed.
