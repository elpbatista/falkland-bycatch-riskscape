"""Generate an H3 grid for the configured region."""

import math
from pathlib import Path

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from riskscape.config import cfg, paths


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


polygon = {
    "type": "Polygon",
    "coordinates": [[
        [xmin_b, ymin_b],
        [xmax_b, ymin_b],
        [xmax_b, ymax_b],
        [xmin_b, ymax_b],
        [xmin_b, ymin_b],
    ]],
}


resolution = cfg["grid"]["resolution"]

shape = h3.geo_to_h3shape(polygon)
hex_ids = h3.h3shape_to_cells(shape, resolution)

records = []

for cell_id in hex_ids:
    boundary = h3.cell_to_boundary(cell_id)
    geometry = Polygon([(lng, lat) for lat, lng in boundary])
    lat, lon = h3.cell_to_latlng(cell_id)

    records.append({
        "hex_id": cell_id,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "geometry": geometry,
    })


crs = cfg["region"]["crs"]
region_name = cfg["region"]["name"]

gdf = gpd.GeoDataFrame(records, crs=crs)

output_dir = paths["grids"]
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"h3_res{resolution}_{region_name}.geojson"

gdf.to_file(output_file, driver="GeoJSON")

print("Grid saved:", output_file)
print("Hex cells:", len(gdf))