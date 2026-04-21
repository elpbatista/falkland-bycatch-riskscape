"""H3 grid utilities for the riskscape package."""

import math
import logging

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from riskscape.config import cfg, paths

logger = logging.getLogger(__name__)


def build_h3_grid() -> None:
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
            "h3_index": cell,
            "lat": lat,
            "lon": lon,
            "geometry": geometry,
        })

    gdf = gpd.GeoDataFrame(records, crs=cfg["region"]["crs"])
    output_dir = paths["grids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    region_name = cfg["region"]["name"]
    base_name = f"h3_res{resolution}_{region_name}"

    geojson_file = output_dir / f"{base_name}.geojson"
    gpkg_file = output_dir / f"{base_name}.gpkg"
    parquet_file = output_dir / f"{base_name}.parquet"

    gdf.to_file(geojson_file, driver="GeoJSON")
    gdf.to_file(gpkg_file, driver="GPKG")
    gdf.to_parquet(parquet_file, engine="pyarrow", index=False)

    for fname in (geojson_file, parquet_file, gpkg_file):
        logger.info(f"Grid saved: {fname}")
    logger.info("Hex cells: %d", len(gdf))


def _main() -> None:
    build_h3_grid()


if __name__ == "__main__":
    _main()
