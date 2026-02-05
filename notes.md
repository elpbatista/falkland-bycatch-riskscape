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
