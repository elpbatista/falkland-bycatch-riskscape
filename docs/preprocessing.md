# Preprocessing Pipeline — Final Specification

## 1. Load raw datasets

- Read all datasets from `data/raw`
- No transformations at this stage

## 2. Build dataset-specific lookups

- Create spatial mapping from source grid → H3 cells
- One lookup per dataset

## 3. Aggregate to H3 grid

- Apply lookup
- Aggregate raw values to H3 cells
- Output keyed by:
  - `h3`, `date`, variable

## 4. Temporal alignment

- Ensure all datasets use:
  - `date` at daily resolution
- No gap handling defined yet

## 5. Schema / type normalization

Apply the fixed schema:

| Column    | Type           |
|-----------|----------------|
| `date`    | datetime64[ns] |
| `h3`      | uint64         |
| variables | float32        |

## 6. Derived variables

### 6.1 Gradients

- Computed on aggregated data
- Spatial (H3-based)
- Per variable

### 6.2 Anomalies

- Computed on aggregated data
- Temporal baseline (not further specified yet)
- Per variable

## 7. Seasonal encoding (computed once)

### Build seasonal lookup

- `adjusted_doy | doy_sin | doy_cos`

### Rule

- Compute DOY from `date`
- If leap year and DOY > 59 → subtract 1

### Then

- Map `adjusted_doy` → `doy_sin`, `doy_cos`

## 8. Merge datasets

Merge all variables into one table:

- `h3 | date | sst | chl | ssh | ...`

## 9. Attach seasonal encoding

- Compute `adjusted_doy` from `date`
- Join:
  - `adjusted_doy → doy_sin, doy_cos`

## 10. Inspect aggregated data

- Check distributions
- Check ranges
- Check missing values

## 11. Optional transformations (deferred)

- Example: `chl → log10`
- Applied only after inspection

## 12. Standardization

- Apply after all transformations
- Per variable

## 13. Save output

- Columnar format (Parquet)
- Final structure:

  - `h3 | date | doy_sin | doy_cos | variables...`

---

## Key decisions

- Spatial key: `h3` (uint64)
- Temporal key: `date`
- Seasonal encoding: adjusted DOY → sin/cos
- Aggregation occurs before any transformation
- `chl` transformation: deferred
- Gradients and anomalies included in preprocessing
- Seasonal encoding computed once (365-day cycle)
- Format: Parquet
- Compression: ZSTD
- Storage layout: partitioned by year

---

## Characteristics

- No duplicated seasonal computation
- No per-year lookup tables
- No assumptions about transformations before inspection
- Dataset-specific logic limited to lookup and aggregation
