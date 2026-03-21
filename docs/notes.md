# Notes

## Study Area Boundings

```Text
Falkland Islands study area
CRS: EPSG:4326
Longitude: -64 to -51
Latitude:  -57 to -47
---
xmin = -64
ymin = -57
xmax = -51
ymax = -47
```

```Python
bbox = [-64, -57, -51, -47]
from shapely.geometry import box
falklands_bbox = box(-64, -57, -51, -47)
```

```JSON
{
  "type": "Polygon",
  "coordinates": [
    [
      [-64, -57],
      [-51, -57],
      [-51, -47],
      [-64, -47],
      [-64, -57]
    ]
  ]
}
```

## The architecture (Prototype)

- Internal grid
  - Type: square
  - Size: ~5 km × 5 km
  - Projection: equal-area
  - Planned evolution: square → hex (optional)
- Temporal resolution
  - Daily (UTC)
- Risk computation
  - Per internal cell per day
- Reporting / management
  - Aggregate to licence squares (mean + max)

> Bycatch risk was computed daily on a ~5 km equal-area square grid and subsequently aggregated to Falkland Islands fishing licence squares for management-scale interpretation.

An equal-area projection ensures:

- Each cell represents the same spatial exposure
- Risk values are comparable across the domain
- Aggregation math actually means what you think it means

## Design note — future grid refinement

Future refinement: Once risk signal stability is established and spatial smoothness becomes a priority, evaluate switching the internal grid to hexagonal cells of comparable area.

Licence squares will remain the reporting and management unit, regardless of internal grid geometry.

Grid shape is intentionally decoupled from risk logic to allow this switch without refactoring core model

## What “fishing effort” actually means

At daily resolution, effort can mean several things:

- Vessel presence
- Vessel time spent
- Number of operations (sets, hauls, nights)
- AIS-derived activity density
- Observer-reported effort units

For bycatch risk, effort is not just a covariate; it’s the exposure term.

No effort → no bycatch risk, no matter how good the habitat looks.

### Time-weighted fishing effort per cell per day

Specifically: **Total fishing hours per ~5 km cell per day**. This is my exposure layer.

`effort_hours[cell, day] = Σ Δt`

> Time-weighted fishing effort (hours) per ~5 km equal-area grid cell per day, derived from AIS/VMS fishing-activity classifications.

### The risk logic

`Risk = Exposure × Susceptibility`

Where:

- Exposure = fishing effort (hours)
- Susceptibility = environmental + biological risk

> Daily bycatch risk is defined as realized risk, conditioned on fishing effort; latent risk surfaces may be computed separately for exploratory or forecasting purposes.

### Risk decomposition

> Bycatch risk is decomposed into a latent (hazard) component driven by environmental and biological conditions, and a realized (impact) component obtained by conditioning latent risk on fishing effort.

`realized_risk = latent_risk × effort`

**Latent risk (hazard):** The environmentally and biologically driven potential for bycatch to occur in a given place and day, independent of fishing effort.

**Realized risk (impact):** The bycatch risk actually expressed on a given day, conditioned on fishing effort.

```Text
Meteorology + Satellite
        ↓
Continuous Seascape Gradients (daily, 5 km)
        ↓
Latent Hazard Model (bycatch ~ seascapes)
        ↓
Latent Hazard Surface (cell × day)
        ↓
Realized Risk = Hazard × Effort
        ↓
Licence Square Aggregation

---

Tracking Data
        ↓
Behavior relative to Seascapes and Hazard

---

Environment → Seascapes
        ↓
Latent Hazard (environment-only)
        ↓
Species SDM (environment-only)
        ↓
Species Presence Probability

Species Latent Risk = Hazard × Species Presence

Species Realized Impact = Species Latent Risk × Effort
```

Forecasts operate on the latent layer while operations and reporting operate on realized risk

### How to handle effort in a forecasting context

Once latent risk exists, effort becomes modular:

- Observed effort → retrospective analysis
- Smoothed effort → near-term nowcasts
- Hypothetical effort → “what-if” scenarios
- Zero effort → protected area simulations

### What latent risk from tracking should represent

Tracking-derived hazard should answer:

> How likely is the species to be in this cell on this day?

## Memory notes

> For production pipelines, use PyArrow or DuckDB with explicit schema and batch sizing. Sort by space (Hilbert curve) during the write to maximize predicate pushdown later.

**GAM** = Generalized Additive Model

It is a statistical model that allows nonlinear relationships between predictors (seascape gradients) and a response (latent bycatch hazard), while remaining interpretable.

A GAM is:

A nonlinear regression model that estimates smooth relationships between environmental gradients and latent bycatch hazard, making it ideal for continuous seascape-driven risk modeling on the daily 5 km grid.

---

**The probability that a cell-day under a given environmental regime is prone to bycatch-generating conditions**.

## Conceptual Architecture

Environment → Seascape gradients.  
Seascapes → Probability of bycatch-prone regime.  
Probability → Risk category (Low/Moderate/High).  
Effort → Scales impact, not hazard.  

Tracking → Evaluates whether species use high-probability regimes

---

## Falklands Daily Seascape-Driven Bycatch System — Summary

- Temporal resolution: Daily (UTC)

- Internal modeling grid:
  - ~5 km equal-area cells
  - Square cells (initial implementation)
  - Hexagonal cells (optional refinement)
  - Grid geometry modular and replaceable

- Reporting unit:
  - Licence squares (aggregation only)

- Core structure:
  - Environment → Seascapes → Latent Risk → Realized Impact

- Seascape representation:
  - Continuous environmental gradients
  - No discrete classification in core forecast model
  - Nonlinear embedding optional (Phase 2)

- Environmental inputs (cell × day):
  - Wind metrics
  - SST
  - SST gradient
  - Chlorophyll
  - Chlorophyll gradient
  - SSH (optional)

- Latent bycatch risk:
  - Probability surface
  - Independent of fishing effort
  - Modeled as: `logit(P) = f(seascape gradients)`

- Risk output:
  - Continuous probability (0–1)
  - Categorized into `Low / Moderate / High` after modeling

- Fishing effort:
  - Not used in latent model
  - Used only to compute realized impact
  - `Realized impact = latent probability × effort`

- Tracking data:
  - Not used to define hazard
  - Used to analyze species behavior relative to seascapes and hazard

- Forecasting workflow:
  - Forecast environmental gradients
  - Predict latent probability
  - Categorize risk
  - Optionally scale by projected effort

- Phase strategy:
  - Phase 1: Continuous gradients + nonlinear additive model
  - Phase 2: Nonlinear embedding only if diagnostics justify

> Adding the species layer modularly allows forecasting different species depending on tracking data availability, without altering the core seascape-driven latent hazard framework.

## Five-Layer System

1. Seascapes (Physical Layer)  
   Environment → daily gradients (5 km grid)

2. Species SDM (Ecological Layer)  
   Seascape gradients + tracking data → P(species present)

3. Latent Hazard (Interaction Layer)  
   Seascape gradients + bycatch observations → P(bycatch-prone)

4. Species Latent Risk  
   P(bycatch-prone) × P(species)

5. Realized Impact (Operational Layer)  
   P(bycatch-prone) × P(species) × Effort

---

Credentials file stored in `/Users/pb/.copernicusmarine/.copernicusmarine-credentials`.

---

## Layer 2 — Derived Features (Ecological Feature Engineering)

### What Layer 2 Should Do (Conceptually)

Layer 1 = ocean state
Layer 2 = ocean structure

> I am no longer describing what the ocean is, but how it is organized dynamically. That’s what species respond to.

Layer 2 features are:

- Same domain
- Same spatial grid
- Same temporal resolution
- Always used together in modeling

So they belong together: One Layer 2 table, wide format, split by year (same structure as Layer 1).

```text
date | h3 | sst | chl | ssh | sst_grad | chl_grad | ssh_grad | sst_anom | ...
```

### Why This Is a Big Step

Layer 1 = 13.6M rows of environmental state
Layer 2 = ecological signal extraction

This is where:

- Frontal systems emerge
- Habitat structure appears
- Risk-relevant gradients become visible

This is where the project becomes scientifically interesting.

---

## Note: Gradient Computation Strategy (Layer 2)

### Decision

Gradients for Layer 2 will be computed directly on the H3 grid using neighbor-based differences.

Raster-level gradients are not required, as all environmental variables have already been harmonized and aggregated to the H3 spatial framework in Layer 1.

This ensures architectural consistency and computational scalability.

### Rationale

- All modeling occurs at H3 resolution.
- Sub-H3 raster gradients are unnecessary for the ecological scale of interest.
- Species respond to habitat-scale contrasts, not pixel-scale derivatives.
- H3-based gradients are computationally efficient and scalable across 10 years of daily data.

### Mathematical Definition

For each H3 cell i:

G_i = sqrt( (1 / n) * sum((X_i - X_j)^2) )

Where:

- X_i = value of the environmental variable at cell i
- X_j = value of the same variable at neighboring cell j
- N(i) = set of ring-1 neighboring H3 cells of cell i
- n = number of valid neighboring cells
- G_i = gradient magnitude at cell i

This represents the root-mean-square contrast between a cell and its immediate neighbors.

### Implementation Details

- Neighbor relationships will be precomputed once.
- Gradients computed daily for each variable.
- Output stored as float32.
- Gradients aligned with the master H3 grid.
- One wide Layer 2 table per year.

### Interpretation

Gradient magnitude represents habitat contrast intensity
(e.g., thermal front strength).

These features will serve as primary ecological structure predictors
in the bycatch risk model.

Status: Confirmed strategy for Layer 2 implementation.

## Gradient Recommendation

- SST gradient → raw
- SSH gradient → raw
- Chlorophyll gradient → log-transform first

### Why log10 Is Better Here

Because oceanography often thinks in orders of magnitude.

Example:

- 0.1 mg m⁻³ → oligotrophic
- 1.0 mg m⁻³ → productive
- 10 mg m⁻³ → bloom

These are powers of 10.

So:

```text
log10(0.1) = -1
log10(1)   =  0
log10(10)  =  1
```

Very intuitive.

A 1-unit difference in log10:

= 10× concentration change

That’s ecologically meaningful.

## Wind Variables (ERA5) — Quick Reference

### Components

- **u10**: zonal wind (east–west)  
  - > 0 → toward east  
  - < 0 → toward west  

- **v10**: meridional wind (north–south)  
  - > 0 → toward north  
  - < 0 → toward south  

Both are measured at **10 m above surface** and in **m/s**.

---

### Wind Speed (Layer 2C primary variable)

$$
WS = \sqrt{u_{10}^2 + v_{10}^2}
$$

- Scalar magnitude  
- Represents wind intensity  
- Typical range: ~0–20 m/s  

Stored as:

`wind`

---

### Wind Direction (optional, later)

$$
\theta = \arctan2(v_{10}, u_{10})
$$

Convert to meteorological convention:

$$
\theta_{met} = (270 - \theta_{deg}) \bmod 360
$$

- Degrees (0–360)  
- Indicates **where wind comes FROM**

---

### Wind Notes

- No additional data needed — $u10$ and $v10$ are sufficient  
- Direction is **not required** for initial modeling  
- It can be computed later from the same raw ERA5 files if needed  
- Wind speed is the primary variable for:
  - mixing  
  - fronts  
  - ecological forcing  

```text
         v (north)
          ↑
          |
          |
          • ----→ u (east)
```
