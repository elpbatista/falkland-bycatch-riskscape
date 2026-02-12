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
Environmental + biological data
          ↓
Latent risk (cell, day)
          ↓
   × Fishing effort (observed or hypothetical)
          ↓
Realized risk (cell, day)
          ↓
Aggregate to licence squares
```

Forecasts operate on the latent layer while operations and reporting operate on realized risk

### How to handle effort in a forecasting context

Once latent risk exists, effort becomes modular:

- Observed effort → retrospective analysis
- Smoothed effort → near-term nowcasts
- Hypothetical effort → “what-if” scenarios
- Zero effort → protected area simulations

## Memory notes

> For production pipelines, use PyArrow or DuckDB with explicit schema and batch sizing. Sort by space (Hilbert curve) during the write to maximize predicate pushdown later.
