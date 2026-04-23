"""Grid I/O utilities."""

from pathlib import Path

import geopandas as gpd

from riskscape.config import cfg, paths


def load_grid(uint64: bool = False) -> gpd.GeoDataFrame:
    """Load the H3 grid."""
    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    suffix = "_uint64" if uint64 else ""
    grid_file = f"h3_res{resolution}_{region_name}{suffix}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    grid = gpd.read_parquet(grid_path)

    required = {"h3", "lat", "lon", "geometry"}
    missing = required - set(grid.columns)
    if missing:
        raise ValueError(f"Grid schema invalid, missing: {missing}")

    return grid
