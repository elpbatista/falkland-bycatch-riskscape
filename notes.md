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
