from pathlib import Path
import yaml
import math

import geopandas as gpd
from shapely.geometry import Polygon

import h3


CONFIG_FILE = "config/config.yaml"


# ------------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------------

with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)

bbox = cfg["region"]["bbox"]
buffer_km = cfg["region"]["buffer_km"]
resolution = cfg["grid"]["resolution"]

xmin = bbox["xmin"]
ymin = bbox["ymin"]
xmax = bbox["xmax"]
ymax = bbox["ymax"]


# ------------------------------------------------------------------
# Convert buffer from km to degrees
# ------------------------------------------------------------------

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

hex_ids = h3.polyfill_geojson(polygon, resolution)

records = []

for h in hex_ids:

    boundary = h3.h3_to_geo_boundary(h, geo_json=True)
    poly = Polygon(boundary)

    lat, lon = h3.h3_to_geo(h)

    records.append({
        "hex_id": h,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "geometry": poly
    })


# ------------------------------------------------------------------
# Save grid
# ------------------------------------------------------------------

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

output_dir = Path(cfg["paths"]["grids"])
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"h3_res{resolution}_falklands.geojson"

gdf.to_file(output_file, driver="GeoJSON")

print("Grid saved to:", output_file)
print("Number of hex cells:", len(gdf))