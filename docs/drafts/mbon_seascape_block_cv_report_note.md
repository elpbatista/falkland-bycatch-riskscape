# NOAA/MBON Seascapes and Block-CV Validation Draft

This note documents the NOAA/MBON seascape work added as an alternative to the project-specific KMeans seascape classes. It is intended as report-ready draft text, but it is kept in the main repository notes rather than the report repository.

## Motivation

The first seascape experiment used feature-only KMeans classes generated from the project environmental feature grid. Those classes were useful for interpretation, but they were internally defined and sensitive to aggregation choices. To make the seascape component more externally grounded, we tested the NOAA/AOML MBON dynamic seascape product as an operational seascape layer.

The goal was not to replace the continuous environmental predictors. Instead, the MBON classes were used as an additional blocked-validation structure, allowing the model to be tested across externally defined environmental regimes.

## NOAA/MBON Seascape Source

The NOAA/AOML MBON product provides dynamic pelagic seascape classes derived from satellite oceanographic conditions. The product used here was the 8-day seascape dataset served through NOAA/AOML THREDDS:

`https://cwcgom.aoml.noaa.gov/thredds/dodsC/SEASCAPE_8DAY/SEASCAPES.nc`

The variables used were:

- `CLASS`: dominant MBON seascape class.
- `P`: class probability or confidence field.
- `time`, latitude, and longitude coordinates.

Because the MBON product is 8-day rather than daily, each project date was assigned to the nearest available MBON source date. The output table retains both the daily project `date` and the original `mbon_source_date`, plus `days_from_mbon_source` to make the temporal offset explicit.

## H3 Assignment

MBON seascapes were assigned to the canonical project grid: the Falkland fisheries-grid bounding box plus a 50 km buffer, represented as H3 resolution 6 cells. The resulting grid contains 37,209 H3 cells.

The assignment table was generated for the full 2014-2023 project period, so it is ready for future telemetry additions beyond the current 2022-2023 species observations. The output table is:

`data/modeling/mbon_seascapes_8day_area_weighted/year=<year>/part.parquet`

The aggregation method follows the same area-weighted raster-to-H3 logic used by the environmental feature pipeline. Raster pixels were converted to pixel geometries, intersected with H3 cell polygons, and weighted by geodesic overlap area. For each H3 cell and MBON source date, the dominant MBON class was selected by summed overlap weight. The table includes:

- `h3`
- `date`
- `mbon_seascape`
- `mbon_probability`
- `mbon_class_weight`
- `mbon_source_date`
- `days_from_mbon_source`

Dates were stored as timezone-free daily timestamps to remain consistent with the rest of the modeling pipeline.

## Use in Block-CV

The block-CV training script now supports:

`--split environmental_mbon_seascape`

The first implementation held out whole MBON seascape classes. This was too coarse because several MBON classes covered very broad areas and produced highly imbalanced folds. In one single-holdout run, the test fraction was approximately 49%, and a 5-fold diagnostic produced fold sizes ranging from roughly 1% to 49% of the data.

To fix this, the validation groups now combine MBON class with spatial H3 parent blocks. Each row is assigned a group of the form:

`mbon_seascape_<class>__<h3_parent_res4>`

This preserves environmental separation while avoiding folds dominated by one large seascape class. It also keeps spatial structure in the validation design, which is more consistent with block-CV principles.

## Split Balance Result

After crossing MBON seascapes with spatial blocks, the 5-fold split became well balanced overall and by species.

| Fold | BBAL test rate | SAFS test rate |
|---:|---:|---:|
| 1 | 20.37% | 19.83% |
| 2 | 20.01% | 20.13% |
| 3 | 19.68% | 20.13% |
| 4 | 20.06% | 19.92% |
| 5 | 19.88% | 19.99% |

The overall mean test fraction was 0.200 with a standard deviation of 0.00058.

## Model Result

The 5-fold Extra Trees run using the MBON-spatial grouped split produced the following summary:

| Metric | Mean | Std. dev. |
|---|---:|---:|
| `r2` | 0.231 | 0.245 |
| `rmse` | 74.287 | 106.436 |
| `mae` | 5.002 | 4.840 |
| `r2_log` | 0.512 | 0.041 |
| `rmse_log` | 0.557 | 0.041 |
| `mae_log` | 0.359 | 0.018 |

The raw-scale metrics were unstable because a small number of high residence-index observations strongly affected some folds. The log-scale metrics were more stable and are more appropriate for comparing this model because the target is trained on `log1p(ResidenceIndex)`.

For context, the existing KMeans K15 5-fold split had `r2_log_mean = 0.491`, while the MBON-spatial split had `r2_log_mean = 0.512`. This suggests that the externally defined MBON seascape split is at least competitive with the internal KMeans seascape split on the log-transformed target, while also being more interpretable as an external environmental-regime validation.

## Interpretation

The NOAA/MBON seascape product is useful as an externally defined environmental regime layer. However, using whole seascape classes as validation folds is too coarse for this study area because the classes can occupy very large spatial extents. Combining MBON class with spatial parent blocks produces a better validation design: it retains environmental structure while keeping train/test partitions balanced.

The current result supports using MBON-spatial grouped cross-validation as an additional robustness check for species-use modeling. It should be described as a validation and sensitivity-analysis tool rather than as a replacement for the continuous environmental predictors or the hybrid species-use model.

## Implementation Notes

Relevant files:

- `scripts/build_mbon_seascape_assignments.py`
- `src/riskscape/model/block_cv_train.py`

Relevant outputs:

- `data/modeling/mbon_seascapes_8day_area_weighted/`
- `data/modeling/metrics/species_model_environmental_mbon_seascape_mbon_spatial_5fold_split_diagnostics.csv`
- `data/modeling/metrics/species_model_environmental_mbon_seascape_mbon_spatial_5fold_block_cv_metrics.csv`
- `data/modeling/metrics/species_model_environmental_mbon_seascape_mbon_spatial_5fold_block_cv_summary.csv`

