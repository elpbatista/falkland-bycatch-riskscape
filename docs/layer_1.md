# Layer 1 Specification — Seascapes (Physical Layer)

## Overview

Layer 1 defines the **physical ocean environment** used by the Dynamic Bycatch Riskscape Framework.  
It represents the **daily ocean state** using satellite and altimetry observations aggregated to a **hexagonal H3 grid (resolution 6)**.

Layer 1 provides the environmental foundation used by:

- **Layer 2** — Species Distribution Models  
- **Layer 3** — Latent Bycatch Hazard Models  

Layer 1 intentionally excludes biological, fishing, and meteorological variables.

---

## Purpose

Layer 1 provides two environmental descriptors:

1. **Ocean state variables**
2. **Daily seascape gradients**

These represent the **physical and biogeochemical structure of the ocean**, which influences marine ecosystem dynamics and predator–fishery interactions.

---

## Spatial Framework

### Grid System

| Parameter      | Value             |
|----------------|-------------------|
| Grid type      | H3 hexagonal grid |
| Resolution     | 6                 |
| Mean cell area | ~36 km²           |

Each record is indexed by:

- `date`
- `h3_id`

---

## Study Area

Coordinate reference system:

```Text
EPSG:4326
```

Bounding box:

```Text
xmin: -64
ymin: -57
xmax: -51
ymax: -47
```

---

## Processing Buffer

A **50 km buffer** is applied to the study area during processing.

Purpose:

- avoid edge artifacts in gradient computation
- allow neighborhood calculations for H3 cells near boundaries

Final outputs may optionally be clipped back to the original study area.

---

## Temporal Coverage

| Parameter           | Value     |
|---------------------|-----------|
| Training period     | 2014–2023 |
| Temporal resolution | Daily     |

---

## Environmental Inputs (Locked)

Layer 1 uses three environmental variables.

| Variable      | Dataset       | Description                 |
|---------------|---------------|-----------------------------|
| SST           | GHRSST MUR L4 | Sea Surface Temperature     |
| Chlorophyll-a | VIIRS L3 SMI  | Ocean color chlorophyll     |
| SSH (ADT)     | CMEMS         | Absolute Dynamic Topography |

---

## Variable Transformations

### Chlorophyll transformation

Chlorophyll values are log-transformed:

```Text
chl_log10 = log10(chl)
```

Zero and invalid values must follow the mask provided in the source dataset.

---

## Standardization

Variables are standardized using **z-score normalization** computed across the training dataset.

```Text
z = (x - mean) / std
```

Applied variables:

- `sst_z`
- `chl_log10_z`
- `adt_z`

---

## Ocean State Representation

For each H3 cell `h` and day `t`, the ocean state vector is:

```Text
S(h,t) = [sst_z, chl_log10_z, adt_z]
```

Where:

- `h` = H3 cell
- `t` = time (day)

This defines the **continuous environmental state** of the ocean.

---

## Mapping Environmental Data to H3 Cells

Environmental datasets are mapped to H3 cells using **centroid sampling**.

Procedure:

1. Compute centroid coordinates for each H3 cell
2. Sample raster values at the centroid location
3. Assign sampled value to the H3 cell

Centroid sampling is suitable for:

- SST
- SSH
- Chlorophyll

because these variables vary smoothly at the spatial scale of H3 resolution 6.

---

## Seascape Gradient Computation

Gradients represent **environmental fronts and boundaries**.

Because the grid is hexagonal, gradients are estimated using **neighbor differences**.

For each H3 cell `h`:

1. Identify neighboring cells `N(h)` using H3 k=1 neighbors
2. Compute gradient magnitude:

```Text
grad_v(h,t) = mean(| v(h,t) - v(n,t) | for n in N(h))
```

Where `v` represents one environmental variable.

---

## Derived Gradient Variables

The following gradient features are computed:

| Variable | Description                    |
|----------|--------------------------------|
| grad_sst | SST gradient magnitude         |
| grad_chl | Chlorophyll gradient magnitude |
| grad_adt | ADT gradient magnitude         |

These gradients approximate:

- thermal fronts
- productivity boundaries
- mesoscale circulation structures

---

## Layer 1 Outputs

Layer 1 produces two data products.

## Ocean State Dataset

Continuous standardized environmental state.

Columns:

- `date`
- `h3_id`
- `sst_z`
- `chl_log10_z`
- `adt_z`

---

## Gradient Dataset

Derived seascape gradients.

Columns:

- `date`
- `h3_id`
- `grad_sst`
- `grad_chl`
- `grad_adt`

---

## Explicit Exclusions

The following variables are **not part of Layer 1**:

- SLA
- nFLH
- wind variables
- species distribution data
- fishing effort
- bycatch observations

These belong to later layers of the modeling framework.

---

## Conceptual Role in the Model

Layer 1 describes **physical ocean structure only**.

Higher layers use these outputs as predictors:

| Layer   | Use                           |
|---------|-------------------------------|
| Layer 2 | Species distribution modeling |
| Layer 3 | Bycatch hazard modeling       |
| Layer 4 | Species latent risk           |
| Layer 5 | Realized operational impact   |

---

## Summary

Layer 1 defines the daily ocean environment using three standardized variables:

```Text
S(h,t) = [sst_z, chl_log10_z, adt_z]
```

Spatial gradients derived from these variables provide the **seascape structure** used by higher layers of the bycatch riskscape model.

---

Strictly following **Kavanaugh et al. (2014)**

> Continuous implementation, daily, 5 km equal-area grid.  
> 2014–2023 (10-year) training dataset, with 2024–2025 (2-year) forecast period.

## Data Sources

> Resolution mismatch is expected.  
> All variables will be harmonized to 5 km grid.

```Text
data/
   raw/
      sst/
      chlorophyll/
      adt/
```

### Sea Surface Temperature (SST)

Temperature of the ocean’s uppermost layer, representing the thermal state of the surface ocean and a key indicator of water mass structure and stratification.

### Chlorophyll-a (Chl-a)

Satellite-derived proxy for phytoplankton biomass, representing the biogeochemical state and surface ocean productivity.

### Sea Surface Height / Absolute Dynamic Topography (SSH / ADT)

Height of the sea surface relative to a geoid; reflects ocean circulation, mesoscale structure (eddies, fronts), and dynamic pressure gradients.
