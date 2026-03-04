"""Generate an H3 grid for the configured region."""

import math

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from riskscape.config import cfg, paths


def build_h3_grid():
    """Create and save the H3 grid defined in config.yaml."""

    bbox = cfg["region"]["bbox"]
    buffer_km = cfg["region"]["buffer_km"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    mid_lat = (ymin + ymax) / 2

    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * math.cos(math.radians(mid_lat)))

    xmin -= dlon
    xmax += dlon
    ymin -= dlat
    ymax += dlat

    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]],
    }

    resolution = cfg["grid"]["resolution"]

    shape = h3.geo_to_h3shape(polygon)
    cells = h3.h3shape_to_cells(shape, resolution)

    records = []

    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        geometry = Polygon([(lon, lat) for lat, lon in boundary])
        lat, lon = h3.cell_to_latlng(cell)

        records.append({
            "id": cell,
            "lat": lat,
            "lon": lon,
            "geometry": geometry,
        })

    gdf = gpd.GeoDataFrame(records, crs=cfg["region"]["crs"])

    output_dir = paths["grids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    region = cfg["region"]["name"]
    output_file = output_dir / f"h3_res{resolution}_{region}.geojson"

    gdf.to_file(output_file, driver="GeoJSON")

    print("Grid saved:", output_file)
    print("Hex cells:", len(gdf))


if __name__ == "__main__":
    build_h3_grid()