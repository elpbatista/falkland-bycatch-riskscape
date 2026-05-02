# Layer 2A — Gradient-Based Physical Seascape Features

## Purpose

Layer 2A derives spatial contrast metrics from the continuous physical
state produced in Layer 1. These gradients represent dynamic ocean
structure, including front strength and mesoscale variability, on the
H3 grid.

Layer 2A transforms environmental fields into physically interpretable
contrast indicators suitable for ecological modeling.

## Inputs

From Layer 1 (per year):

- `sst` — Sea Surface Temperature (Kelvin)
- `chl` — Chlorophyll-a (mg m^-3)
- `ssh` — Sea Surface Height / Absolute Dynamic Topography (m)

Spatial reference:

- H3 grid (resolution 6, Falkland Islands region)
- Ring-1 neighbor topology (indexed graph)

## Mathematical Definition

For each H3 cell $i$ and variable $X$:

$$
G_i = \sqrt{\frac{1}{n} \sum_{j \in N(i)} (X_i - X_j)^2}
$$

Where:

- ($N(i)$) = set of ring-1 neighboring H3 cells of cell $i$
- $(n)$ = number of valid neighboring cells
- $(X_i)$ = value of the variable at cell $i$
- $(X_j)$ = value of the same variable at neighboring cell $j$
- $(G_i)$ = root-mean-square spatial contrast at cell $i$

This represents the local gradient magnitude, or front intensity, around
each H3 cell.

## Variable-Specific Notes

### SST Gradient (`sst_grad`)

- SST is converted from Kelvin to Celsius before gradient computation.
- Units: degrees C contrast across neighboring cells.

### Chlorophyll Gradient (`chl_grad`)

- Log transform applied: `log10(chl)`
- Gradient computed in log space.
- Interpreted as relative productivity contrast.

### SSH Gradient (`ssh_grad`)

- Computed directly in meters.
- Represents mesoscale elevation contrast.

## Output Structure

One file per year:

`data/layer2/year=YYYY.parquet`

Columns:

- `date`
- `h3`
- `sst`
- `chl`
- `ssh`
- `sst_grad`
- `chl_grad`
- `ssh_grad`

Rows per year:

- ~37,209 cells x 365 days
- ~13.5 million rows

## Interpretation

Layer 2A captures:

- Thermal fronts
- Productivity boundaries
- Mesoscale structure intensity

These gradients serve as physical habitat-structure proxies and provide
the first derived dynamic predictors for ecological overlap and bycatch
risk modeling.

## Architectural Position

Layer 2A extends:

- Layer 1: continuous physical state
- Layer 2A: spatial contrast structure

Future Layer 2 extensions may include:

- Seasonal anomalies
- Wind persistence metrics
- Front-wind interaction terms

## Status

Layer 2A is complete and frozen across the full 10-year period.
