# Bycatch Riskscape Pipeline — Current Status

This document summarizes the architecture and components implemented so far for the bycatch riskscape project.

## Project Goal

Build a reproducible pipeline to generate dynamic bycatch risk maps by integrating:

- oceanographic conditions
- species distribution
- fishing effort

The model is structured as a **five-layer system**.

## Five-Layer Model

1. **Seascapes (Physical Layer)**  
   Daily environmental state of the ocean.

2. **Species SDM (Ecological Layer)**  
   Probability of species presence.

3. **Latent Hazard (Interaction Layer)**  
   Probability of bycatch-prone conditions.

4. **Species Latent Risk**  
   Interaction of species presence and hazard.

5. **Realized Impact (Operational Layer)**  
   Risk weighted by fishing effort.

Current development focuses on **Layer 1 (Seascapes)**.

## Study Area

Region: Falkland Islands

CRS: EPSG:4326

Bounding box:

```text
xmin = -64
ymin = -57
xmax = -51
ymax = -47
```

Processing includes a **50 km buffer** around the study area.

## Spatial Grid

Grid system: H3  
Resolution: 6

The grid is generated once and stored as:

```text
data/grids/h3_res6_falkland_islands.geojson
```

Each cell contains:

- `hex_id`
- `centroid_lat`
- `centroid_lon`
- geometry

## Time Range

Training dataset:

```text
2014–2023
```

Future forecasting period:

```text
2024–2025
```

## Environmental Variables (Layer 1)

Layer 1 represents a **continuous multivariate ocean state** using:

| Variable    | Dataset                           |
|-------------|-----------------------------------|
| SST         | GHRSST MUR L4                     |
| Chlorophyll | VIIRS L3 SMI                      |
| SSH         | CMEMS Absolute Dynamic Topography |

Explicit exclusions:

- no SLA
- no nFLH
- no wind
- no clustering outputs

Chlorophyll will be **log10 transformed**.

## Configuration System

All parameters are defined in:

```text
config.yaml
```

Configuration includes:

- region definition
- grid parameters
- time range
- dataset definitions
- project paths

Code accesses configuration via:

```python
from riskscape.config import cfg, paths
```

## Project Structure

```text
bycatch-riskscape/

config.yaml

src/riskscape/
    config.py
    download/
        base.py
        datasets.py
        metadata.py

scripts/
    generate_h3_grid.py
    download_dataset.py

data/
    raw/
    processed/
    grids/

logs/
```

## Grid Generation

Script:

```text
scripts/generate_h3_grid.py
```

Features:

- reads configuration
- applies buffer to bounding box
- generates H3 grid
- stores GeoJSON grid file

## Dataset Downloader

A generic downloader supports all datasets.

Core module:

```text
src/riskscape/download/base.py
```

Features:

- config-driven time range
- parallel downloads
- resumable downloads
- dataset metadata recording
- progress reporting

### Manifest

Each dataset keeps a manifest file:

```text
logs/<dataset>_manifest.csv
```

This allows interrupted downloads to resume automatically.

### Metadata

Each dataset stores provenance information:

```text
data/raw/<dataset>/metadata.json
```

Metadata includes:

- dataset name
- provider
- product version
- variable
- download date
- time range
- source URL

## Dataset Registry

Datasets are defined in:

```text
src/riskscape/download/datasets.py
```

This allows downloading datasets with one command.

Example:

```text
python scripts/download_dataset.py sst
```

## Code Style

The project follows a **minimal Python style**:

- short PEP-257 docstrings
- minimal comments
- no decorative separators
- config-driven parameters
- readable top-to-bottom code

A style guide is documented in:

```text
docs/code_style.md
```

## Current Pipeline

```text
config.yaml
      │
      ▼
generate_h3_grid.py
      │
      ▼
download_dataset.py <dataset>
      │
      ▼
raw environmental data
```

## Next Step

Next component to implement:

```text
build_layer1.py
```

This step will:

```text
raw NetCDF datasets
        ↓
sample to H3 grid
        ↓
daily ocean state table
```

The result will be the **Layer 1 seascape dataset** used by downstream models.
