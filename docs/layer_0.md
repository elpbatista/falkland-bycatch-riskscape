# Layer 0 — Spatial Framework

Layer 0 defines the spatial foundation of the bycatch riskscape system. It establishes the study region, spatial indexing scheme, and the reproducible grid generation process that all subsequent layers will use.

## Architecture

```text
config.yaml
     │
     ▼
scripts/build_grid.py
     │
     ▼
src/riskscape/grid/h3_grid.py
     │
     ▼
data/grids/h3_res6_falkland_islands.*
```

## Study Region

The study region corresponds to the Falkland Islands area and is defined in the project configuration.

CRS: EPSG:4326  
Bounding box: xmin -64, ymin -57, xmax -51, ymax -47  
Buffer applied to the bounding box: 50 km  

The buffer ensures that environmental data near the study area boundary are included during later extraction steps and helps avoid edge artifacts.

## Spatial Grid System

The spatial framework uses the H3 hierarchical hexagonal grid.

Grid system: H3  
Resolution: 6  

At resolution 6, the approximate hexagon area is about 36 km². This resolution provides a good balance between ecological relevance, compatibility with oceanographic datasets, and computational efficiency.

## Grid Generation

The grid is generated programmatically from the configuration file.

Execution command:

`python scripts/build_grid.py`

Pipeline structure:

config.yaml → scripts/build_grid.py → src/riskscape/grid/h3_grid.py → data/grids/

The grid generation process performs the following steps:

1. Read the region definition from config.yaml.
2. Apply the geographic buffer to the bounding box.
3. Construct the buffered polygon.
4. Generate H3 cells covering the polygon.
5. Convert H3 boundaries to polygon geometries.
6. Compute centroid coordinates for each hexagon.
7. Export the resulting grid.

## Grid Schema

Each grid cell contains the following attributes:

```text
id — H3 cell identifier  
lat — centroid latitude  
lon — centroid longitude  
geometry — hexagon polygon  
```

The H3 identifier acts as the spatial index used throughout the modeling pipeline.

## Output Formats

The grid is exported in multiple formats to support both GIS visualization and analytical workflows.

Output directory: data/grids/

Files produced:

```text
h3_res6_falkland_islands.geojson  
h3_res6_falkland_islands.gpkg  
h3_res6_falkland_islands.parquet 
```

GeoJSON provides universal interoperability and is convenient for inspection and debugging.  
GeoPackage offers reliable visualization in GIS software such as QGIS.  
Parquet provides a high-performance columnar format suitable for analytical workflows in Python.

## Reproducibility

The grid can be regenerated deterministically from the configuration by running the grid build script. All spatial parameters are controlled through config.yaml, ensuring that the spatial framework remains reproducible and version-controlled.

## Role in the Modeling System

Layer 0 provides the spatial scaffold for all other components of the riskscape model. All datasets and model outputs are indexed using the H3 cell identifier defined in this layer.

Conceptual structure of the modeling system:

Layer 0 — Spatial grid  
Layer 1 — Physical ocean state (SST, chlorophyll, SSH)  
Layer 2 — Species distribution models  
Layer 3 — Latent bycatch hazard  
Layer 4 — Species-specific risk  
Layer 5 — Realized operational impact

Each layer attaches additional information to the same spatial grid defined in Layer 0.

## Status

Layer 0 is complete. The project now has a fully defined spatial framework that supports environmental data extraction, ecological modeling, and spatial risk analysis.
