# Layer 2B — Anomaly Fields (Temporal Departures)

Layer 2B introduces anomaly fields that quantify temporal deviations from the local seasonal climatology for each H3 cell. While Layer 2A captured spatial structure through gradients, Layer 2B captures interannual and sub-seasonal variability. Together, they describe both spatial contrast and temporal departure.

## Climatology Definition

For each H3 cell $i$ and day-of-year $d$:

$$
\overline{X}_{i,d} =
\frac{1}{K}
\sum_{k=1}^{K} X_{i,d}^{(k)}
$$

Where:

- $X$ = environmental variable (SST, Chlorophyll-a, SSH)
- $d$ = day-of-year (1–366)
- $k$ = year index
- $K$ = total number of years (10 in this dataset)
- $\overline{X}_{i,d}$ = daily climatological mean for cell $i$

Climatology is computed across the full 2014–2023 period using daily (DOY) aggregation per H3 cell.

## Anomaly Definition

For each H3 cell $i$ and time $t$:

$$
X_{i,t}^{anom} = X_{i,t} - \overline{X}_{i,d(t)}
$$

Where:

- $d(t)$ = day-of-year corresponding to time $t$
- $X_{i,t}^{anom}$ = anomaly value at cell $i$

Anomalies represent departures from the expected seasonal baseline at each location.

## Variables Computed

The following anomaly fields were merged directly into the existing Layer 2 yearly tables:

- `sst_anom`
- `chl_anom`
- `ssh_anom`

Each variable is stored as `float32` and written into:

```text
data/layer2/year=YYYY.parquet
```

No separate anomaly directory is maintained. The Layer 2 table remains a single wide feature table per year.

## Chlorophyll Treatment

Chlorophyll was log10-transformed in Layer 1 prior to anomaly computation. Therefore:

- `chl_anom` represents anomaly in log space
- This corresponds to multiplicative deviation in original concentration units
- No epsilon adjustment was required (no zero values detected in the dataset)

## Sanity Check Summary

- The full 10-year anomaly mean is approximately zero (as expected by construction)
- Individual years may exhibit positive or negative mean anomalies, reflecting interannual variability
- Standard deviations are physically realistic for SST, chlorophyll, and SSH
- No row inflation or merge inconsistencies were detected

## Conceptual Role in the Architecture

Layer 2 now contains two orthogonal feature classes:

- Spatial contrast (Layer 2A — gradients)
- Temporal deviation (Layer 2B — anomalies)

Gradients describe persistent or transient front intensity.  
Anomalies describe deviations from seasonal expectation.  

Together, they provide a physically interpretable dynamic seascape representation, forming the foundation for subsequent wind forcing integration (Layer 2C) and ecological interaction modeling.
