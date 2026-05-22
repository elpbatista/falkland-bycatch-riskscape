# Zenodo Data Bundle

Zenodo concept DOI, latest version:
<https://doi.org/10.5281/zenodo.20334806>

Recommended citation DOI:
<https://doi.org/10.5281/zenodo.20337229>

This page documents the released data bundle associated with repository release
`v0.1.0`. The bundle contains derived products, selected figures, and model
outputs that are useful for reproducing the Falkland Islands case study without
placing large generated artifacts in Git.

Raw provider downloads under `data/raw/` are not included. Download or restore
raw inputs through the provider-specific instructions in `docs/datasets.md` and
`docs/authentication.md`.

## Released Files

### `README.md`

Size:
`2.9 kB`.

Description:
archive-level README describing the bundle contents and restoration pattern.

### `falkland-bycatch-riskscape-grids-v0.1.0.tar.gz`

Size:
`17.6 MB`.

Restore to:
`data/grids/`.

Description:
H3 study grid files and spatial index products.

### `falkland-bycatch-riskscape-processed-v0.1.0.tar.gz`

Size:
`116.4 MB`.

Restore to:
`data/processed/`.

Description:
lookup tables, neighbor tables, seasonal lookup products, and other processed
indices used by feature builders and QA checks.

### `falkland-bycatch-riskscape-features-v0.1.0.tar.gz`

Size:
`4.8 GB`.

Restore to:
`data/features/`.

Description:
yearly environmental, fishing-effort, and species-use feature partitions.

### `falkland-bycatch-riskscape-modeling-v0.1.0.tar.gz`

Size:
`11.7 GB`.

Restore to:
`data/modeling/`.

Description:
model-ready tables, selected model artifacts, environmental regimes,
plausibility products, predictions, weekly operator products, validation
metrics, and seascape-conditioned exploratory products.

### `falkland-bycatch-riskscape-plot_exports-v0.1.0.tar.gz`

Size:
`66.0 MB`.

Restore to:
`data/plot_exports/`.

Description:
tabular plot-support products and diagnostics used by plotting scripts.

### `falkland-bycatch-riskscape-plots-v0.1.0.tar.gz`

Size:
`580.2 MB`.

Restore to:
`plots/`.

Description:
static figures, diagnostics, operational maps, and presentation outputs. Some
selected static PNGs are also tracked directly in Git because the public
notebooks display them.

## Restore Layout

After downloading and extracting the bundle files, the repository should have
this local structure:

```text
falkland-bycatch-riskscape/
├── data/
│   ├── grids/
│   ├── processed/
│   ├── features/
│   ├── modeling/
│   └── plot_exports/
└── plots/
```

Extract each archive from the repository root, or copy the extracted folder
contents into the matching destination above.

## Do Not Upload

These products are intentionally excluded from the data bundle.

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

## Notes

The bundle reflects the first public release. Some products can be regenerated
from source data and scripts; others depend on restricted or provider-gated
inputs. The public notebooks describe the role of each product family and the
quality checks used to inspect them.
