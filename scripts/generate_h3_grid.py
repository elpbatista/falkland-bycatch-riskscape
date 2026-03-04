from pathlib import Path
import math

import geopandas as gpd
from shapely.geometry import Polygon

import h3

from riskscape.config import cfg, paths


# ------------------------------------------------------------------
# Convert buffer from km to degrees
# ------------------------------------------------------------------

bbox = cfg["region"]["bbox"]
buffer_km = cfg["region"]["buffer_km"]

xmin = bbox["xmin"]
ymin = bbox["ymin"]
xmax = bbox["xmax"]
ymax = bbox["ymax"]

mid_lat = (ymin + ymax) / 2

dlat = buffer_km / 111.0
dlon = buffer_km / (111.0 * math.cos(math.radians(mid_lat)))

xmin_b = xmin - dlon
xmax_b = xmax + dlon
ymin_b = ymin - dlat
ymax_b = ymax + dlat


# ------------------------------------------------------------------
# Create buffered polygon
# ------------------------------------------------------------------

polygon = {
    "type": "Polygon",
    "coordinates": [[
        [xmin_b, ymin_b],
        [xmax_b, ymin_b],
        [xmax_b, ymax_b],
        [xmin_b, ymax_b],
        [xmin_b, ymin_b]
    ]]
}


# ------------------------------------------------------------------
# Generate H3 grid
# ------------------------------------------------------------------

resolution = cfg["grid"]["resolution"]

shape = h3.geo_to_h3shape(polygon)
hex_ids = h3.h3shape_to_cells(shape, resolution)

records = []

for h in hex_ids:

    boundary = h3.cell_to_boundary(h)
    poly = Polygon([(lng, lat) for lat, lng in boundary])
    lat, lon = h3.cell_to_latlng(h)

    records.append({
        "hex_id": h,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "geometry": poly
    })


# ------------------------------------------------------------------
# Save grid
# ------------------------------------------------------------------

crs = cfg["region"]["crs"]
region_name = cfg["region"]["name"]

gdf = gpd.GeoDataFrame(records, crs=crs)

output_dir = paths["grids"]
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"h3_res{resolution}_{region_name}.geojson"

gdf.to_file(output_file, driver="GeoJSON")

print("Grid saved to:", output_file)
print("Number of hex cells:", len(gdf))