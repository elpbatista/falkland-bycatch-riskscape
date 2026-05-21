# Datasets

This page records source datasets used or referenced by the Falkland Islands
case-study workflow. It is not a complete data release manifest.

Generated data products are not tracked in Git. Public reference layers can be
restored with `scripts/download_reference_data.py`; larger derived data and
plot bundles are intended for Zenodo releases.

## Public Reference Layers

### Falkland Islands Fisheries Grid Squares

Grid squares used as reference units for commercial catches around the Falkland
Islands.

Source:
<https://dataportal.saeri.org/dataset/falkland-islands-fisheries-grid-squares>

Local destination:
`reference/fisheries_grid_squares/`

### Falkland Islands Conservation Zones

Falkland Islands Conservation Zones used for fisheries activities, including
the Falkland Islands Conservation Zone and Falkland Islands Outer Conservation
Zone.

Source:
<https://dataportal.saeri.org/dataset/falkland-islands-conservation-zones>

Local destination:
`reference/ukho_ficz_focz_limits/`

### Natural Earth Land and Coastline

Natural Earth 10m land and coastline layers are used for basemap context.

Source:
<https://www.naturalearthdata.com/>

Local destinations:

- `reference/ne_10m_land/`
- `reference/ne_10m_coastline/`

## Restricted or Provider-Gated Inputs

### Falkland Islands Fisheries Observer Database

Observer data include seabird abundance, interaction records, and seabird and
mammal mortality records. Access may require permission and data-use agreements.

Source:
<https://dataportal.saeri.org/dataset/falkland-islands-fisheries-observer-database>

### Species Telemetry

Species movement observations used by the case study may come from collaborator
or partner datasets. These data should not be redistributed unless access terms
explicitly allow it.

### Global Fishing Watch Fishing Effort

Fishing-effort data are downloaded through Global Fishing Watch APIs. API
access requires a token configured outside Git as `GFW_TOKEN`.

See `docs/authentication.md`.

## Environmental Inputs

The default configuration references environmental inputs such as sea-surface
temperature, chlorophyll-a, sea-surface height, wind, and bathymetry. These
inputs are downloaded or prepared through provider-specific tooling and are
written under ignored `data/` paths.

Bathymetry is cropped from the GEBCO grid through CEDA OPeNDAP and written
under the standard raw dataset folder:

```text
data/raw/bathymetry/
```

GEBCO requests attribution when the grid is used in presentations or
publications; keep the configured product and CEDA archive path visible in
metadata and documentation.

See `config.yaml` and `docs/authentication.md` for provider setup.
