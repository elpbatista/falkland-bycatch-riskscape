# Layer 2C — Wind Variables (ERA5 → H3)

## Overview

Layer 2C integrates near-surface wind data from ERA5 into the H3 spatial grid used in the riskscape system. This layer provides daily wind metrics aligned with the Layer 2 temporal and spatial schema.

Wind variables capture:

- atmospheric forcing intensity
- directional transport
- temporal anomalies
- spatial gradients (fronts and shear zones)

These variables are critical for modeling seabird–fishery interactions and environmental risk patterns in the Falkland Islands region.

---

## Data Source

- Dataset: ERA5 reanalysis (10 m wind components)
- Variables:
  - `u10` — zonal wind (east–west)
  - `v10` — meridional wind (north–south)
- Temporal resolution: daily (aligned to 09:00 UTC)
- Format: NetCDF (monthly files)

---

## Processing Pipeline

### 1. Wind magnitude and components

Wind speed is computed from vector components:

$$
wind = \sqrt{u^2 + v^2}
$$

Where:

- $u = u10$
- $v = v10$

Derived variables:

- `wind` → magnitude (m/s)
- `wind_u` → zonal component
- `wind_v` → meridional component

---

### 2. Wind direction

Wind direction is computed as:

$$
wind\_dir = \left( \frac{180}{\pi} \cdot \arctan2(v, u) \right) \bmod 360
$$

- Units: degrees
- Range: $[0, 360)$

---

### 3. Spatial aggregation (ERA5 → H3)

- ERA5 grid cells are mapped to H3 indices using a pixel → H3 lookup table  
- Aggregation is performed as:

$$
X_{h3} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

Where:

- $X_i$ are ERA5 pixels within an H3 cell
- $n$ is the number of pixels

---

### 4. Temporal alignment

All timestamps are normalized to:

YYYY-MM-DD 09:00:00

This ensures consistency with all Layer 2 variables.

---

### 5. Wind anomaly

Wind anomaly is computed per H3 cell:

$$
wind\_anom = wind - \overline{wind}_{h3}
$$

Where:

- $\overline{wind}_{h3}$ is the mean wind over the full time series for each H3 cell

This isolates temporal deviations from local climatology.

---

### 6. Wind gradient (spatial structure)

Wind gradients are computed using the same method as SST, CHL, and SSH.

For each H3 cell $i$:

$$
G_i = \sqrt{\frac{1}{n} \sum_{j \in N(i)} (X_i - X_j)^2}
$$

Where:

- $N(i)$ = set of ring-1 neighbors (up to 6)
- $n$ = number of valid neighbors

Implementation details:

- Uses precomputed neighbor index table
- Fully vectorized (matrix-based)
- Handles missing values via masking

Result:

- `wind_grad` → spatial variability of wind magnitude

---

## Output Variables

| Variable     | Type    | Description                           |
|--------------|---------|---------------------------------------|
| `wind`       | float32 | Wind speed (m/s)                      |
| `wind_u`     | float32 | Zonal wind component                  |
| `wind_v`     | float32 | Meridional wind component             |
| `wind_dir`   | float32 | Wind direction (degrees)              |
| `wind_anom`  | float32 | Temporal anomaly per H3 cell          |
| `wind_grad`  | float32 | Spatial gradient (RMS over neighbors) |

---

## Data Characteristics (2014–2023)

Typical ranges:

- `wind`
  - mean ≈ 8 m/s
  - range ≈ 0–20 m/s

- `wind_grad`
  - mean ≈ 0.10
  - max ≈ 3–4
  - highlights frontal zones and shear regions

- `wind_anom`
  - mean ≈ 0 (by construction)
  - std ≈ 3 m/s

Missing values:

- ~2.87% (consistent across all wind variables)
- Caused by missing ERA5 coverage after spatial join

---

## Validation

### Spatial consistency

- Wind fields show smooth large-scale patterns
- Gradients reveal coherent structures (not noise)
- No grid artifacts or indexing errors detected

### Temporal consistency

- Stable distributions across years (2014–2023)
- No drift or discontinuities

### Cross-variable alignment

- Wind gradients align structurally with ocean gradients (SST/SSH)
- Suitable for multi-variable riskscape modeling

---

## Notes and Limitations

- Wind gradients are computed on magnitude only  
  - Do not capture directional shear explicitly  

- Yearly mean anomaly maps are not informative  
  - Use daily anomaly fields for analysis  

- ERA5 spatial resolution may smooth fine-scale coastal effects  

---

## Role in Riskscape Modeling

Wind variables contribute to:

- transport processes (odor plumes, prey movement)
- vessel–bird interaction dynamics
- identification of atmospheric fronts and mixing zones

In combination with SST, CHL, and SSH, wind gradients help define:

dynamic environmental boundaries relevant to bycatch risk

---

## File Structure

data/layer2/
  year=2014.parquet
  year=2015.parquet
  ...

Each file contains:

- full H3 grid
- daily records
- all Layer 2 variables including wind
