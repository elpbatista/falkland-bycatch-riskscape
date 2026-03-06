# Layer 1 Specification — Seascapes (Physical Layer)

## 1. Purpose

Layer 1 represents the **physical seascape state** of the Falkland Islands region at daily resolution.

It transforms gridded satellite and reanalysis products into a spatially consistent, H3-indexed dataset suitable for:

- Dynamic ocean management
- Species distribution modeling
- Bycatch risk modeling
- Front detection
- Anomaly detection
- Machine learning pipelines

Layer 1 provides the environmental backbone of the bycatch-riskscape framework.

## 2. Spatial Reference System

### 2.1 Spatial Unit

All variables are aggregated to a fixed:

- **H3 hexagonal grid** — Hierarchical Triangular Mesh provides equal-area cells and efficient spatial indexing
- Resolution: `res = 6`
- Total cells: `37,209`
- Projection: WGS84 (EPSG:4326)

The H3 grid is constructed from:

```text
h3_res{resolution}_{region_name}.geojson
```

This grid is treated as the **master spatial reference**.  
All environmental datasets are aligned to this grid.

## 3. Temporal Resolution

- Daily temporal resolution
- 10-year training dataset
- Timestamp normalized to daily (00:00:00)
- One record per H3 cell per day

Rows per year:

```text
37,209 cells × 365 days ≈ 13.6M rows
```

Total dataset size (10 years):

```text
~136M rows
```

## 4. Environmental Variables

Layer 1 currently includes:

| Variable | Description                 | Units  | Source            |
|----------|-----------------------------|--------|-------------------|
| `sst`    | Sea Surface Temperature     | K      | PO.DAAC MUR       |
| `chl`    | Chlorophyll-a concentration | mg m⁻³ | Copernicus Marine |
| `ssh`    | Absolute Dynamic Topography | m      | Copernicus Marine |

All variables are stored as `float32`.

## 5. Data Processing Workflow

### 5.1 Raw Data (Layer 0)

Data are downloaded from:

- PO.DAAC (MUR SST)
- Copernicus Marine (Chlorophyll, SSH)

All datasets are first cropped to a buffered bounding box of the study region.

### 5.2 Raster → H3 Aggregation

To avoid repeated spatial intersection operations:

1. A lookup table is built per dataset:

   ```text
   pixel_index → H3 index
   ```

2. Aggregation is performed using vectorized bin counting:

   ```text
   mean per H3 cell
   ```

This lookup-based approach reduces extraction cost by **20–50×** compared to polygon intersection at runtime.

### 5.3 Grid Alignment

Because different datasets may cover slightly different pixel extents:

- The H3 grid is treated as the master reference.
- All dataset aggregates are aligned to the grid.
- Missing values are filled with `NaN` (handled during downstream analysis).

Each day therefore contains:

```text
Exactly 37,209 rows (one per H3 cell)
```

## 6. Output Format

Layer 1 is written as Parquet files partitioned by year:

```bash
data/layer1/
    year=2014.parquet
    year=2015.parquet
    ...
```

Schema:

| Column | Type           |
|--------|----------------|
| `date` | datetime64[ns] |
| `h3`   | uint64         |
| `sst`  | float32        |
| `chl`  | float32        |
| `ssh`  | float32        |

Compression:

- Format: Parquet with ZSTD compression
- Approximate total size: **~600–900 MB for 10 years**

## 7. Validation Criteria

Layer 1 is considered valid if:

- Each day contains exactly 37,209 rows.
- No duplicated H3 indices per day.
- All datasets share identical spatial indexing.
- Long-term SST map displays expected shelf–Antarctic gradient.
- Time series shows realistic seasonal cycle.

## 8. Scientific Interpretation

Layer 1 represents the **physical ocean state** and forms the base for:

- Thermal front detection
- Chlorophyll gradient detection
- Eddy detection (SSH gradients)
- SST anomaly computation
- Marine heatwave identification

These derived predictors form Layer 2.

## 9. Design Principles

Layer 1 is:

- **Deterministic** — Reproducible outputs from fixed inputs
- **Config-driven** — All parameters in `config.yaml`
- **Spatially consistent** — All data aligned to master H3 grid
- **Scalable** — Can be extended to new regions by updating config
- **ML-ready** — Normalized, gridded format suitable for feature engineering

It enforces clean separation of concerns:

```text
┌─────────────────────────────────┐
│  Layer 1: Physical seascape     │
│  (SST, Chl, SSH)                │
└──────────────┬──────────────────┘
               │
               ↓
┌─────────────────────────────────┐
│  Layer 2+: Derived features     │
│  (Fronts, anomalies, gradients) │
└──────────────┬──────────────────┘
               │
               ↓
┌─────────────────────────────────┐
│  Bycatch risk model             │
└─────────────────────────────────┘
```

## 10. Known Limitations & Future Extensions

### Current Limitations

- Limited to 3 variables (SST, Chl, SSH)
- No surface wind forcing
- No temporal lag features
- NaN values not interpolated (handled downstream)

### Planned Additions

- **Wind forcing** (u-component, v-component)
- **Gradient magnitude** (SSH and SST gradients)
- **Seasonal climatology** (monthly baseline for anomaly computation)
- **Temporal features** (daily anomalies relative to 10-year climatology)

---

## Summary

Layer 1 converts heterogeneous oceanographic raster products into a unified, H3-indexed daily environmental dataset spanning 10 years at **37,209 cells/day** resolution.

By standardizing spatial indexing and temporal aggregation, Layer 1 provides a **deterministic, reproducible foundation** for downstream feature engineering and bycatch risk modeling.

Key achievement: **~136 million rows** of aligned oceanographic data in <1 GB storage, ready for machine learning pipelines.
