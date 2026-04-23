"""H3 grid utilities for the riskscape package."""

import logging

import geopandas as gpd
import h3
from shapely.geometry import Polygon

from riskscape.config import cfg, paths
from .extent import get_buffered_polygon_geojson

logger = logging.getLogger(__name__)


def build_h3_grid() -> None:
    """Create and save the H3 grid defined in config.yaml."""

    polygon = get_buffered_polygon_geojson()
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

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=cfg["region"]["crs"])

    output_dir = paths["grids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    region_name = cfg["region"]["name"]
    base_name = f"h3_res{resolution}_{region_name}"

    parquet_file = output_dir / f"{base_name}.parquet"
    gdf.to_parquet(parquet_file, engine="pyarrow", index=False)

    print(gdf.head())
    print(gdf.crs)

    logger.info("Grid saved: %s", parquet_file)
    logger.info("Hex cells: %d", len(gdf))


def _main() -> None:
    build_h3_grid()


if __name__ == "__main__":
    _main()