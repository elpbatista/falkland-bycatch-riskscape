"""Build raster -> H3 lookup tables for fast extraction."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from riskscape.config import cfg, paths


def detect_coords(ds: xr.Dataset) -> tuple[str, str]:
    lat_candidates = ("lat", "latitude")
    lon_candidates = ("lon", "longitude")

    lat = next((c for c in lat_candidates if c in ds.coords), None)
    lon = next((c for c in lon_candidates if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError(f"lat/lon coords not found. coords={list(ds.coords)}")

    return lat, lon


def grid_path_from_cfg() -> Path:
    region = cfg["region"]["name"]
    res = int(cfg["grid"]["resolution"])
    return paths["grids"] / f"h3_res{res}_{region}.parquet"


def lookup_dir_from_cfg() -> Path:
    return paths.get("lookups", paths["data"] / "lookups")


def build_lookup(dataset_name: str) -> None:
    dataset_dir = paths["raw"] / dataset_name
    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        print(f"Skipping {dataset_name}: no .nc files in {dataset_dir}")
        return

    ds0 = xr.open_dataset(files[0])
    lat_name, lon_name = detect_coords(ds0)

    lats = ds0[lat_name].values
    lons = ds0[lon_name].values
    ds0.close()

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    pixels = gpd.GeoDataFrame(
        {
            "pixel": np.arange(lat_flat.size, dtype=np.int64),
            "geometry": [Point(xy) for xy in zip(lon_flat, lat_flat)],
        },
        crs="EPSG:4326",
    )

    grid = gpd.read_parquet(grid_path_from_cfg())[["id", "geometry"]]

    joined = gpd.sjoin(pixels, grid, how="left", predicate="within")
    lookup = joined[["pixel", "id"]].rename(columns={"id": "h3"})

    out_dir = lookup_dir_from_cfg()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{dataset_name}_lookup.parquet"
    lookup.to_parquet(out_file, index=False)

    mapped = int(lookup["h3"].notna().sum())
    print(f"{dataset_name}: mapped_pixels = {mapped}")


def main() -> None:
    for name in cfg["datasets"].keys():
        build_lookup(name)


if __name__ == "__main__":
    main()