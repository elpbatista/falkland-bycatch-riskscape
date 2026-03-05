"""Build raster -> H3 lookup tables using direct H3 indexing."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h3

from riskscape.config import cfg


def detect_coords(ds):
    """Detect latitude and longitude coordinate names."""
    lat_candidates = ("lat", "latitude")
    lon_candidates = ("lon", "longitude")

    lat = next((c for c in lat_candidates if c in ds.coords), None)
    lon = next((c for c in lon_candidates if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError(f"lat/lon coordinates not found. coords={list(ds.coords)}")

    return lat, lon


def grid_resolution():
    """Return H3 resolution from config."""
    return int(cfg["grid"]["resolution"])


def raw_dir():
    return Path(cfg["paths"]["raw"])


def lookup_dir():
    path = Path(cfg["paths"]["data"]) / "lookups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_lookup(dataset_name):
    """Create pixel -> H3 lookup table."""

    dataset_dir = raw_dir() / dataset_name
    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        print(f"Skipping {dataset_name}: no .nc files")
        return

    print(f"Building lookup for {dataset_name}")

    ds = xr.open_dataset(files[0])

    lat_name, lon_name = detect_coords(ds)

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ds.close()

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    resolution = grid_resolution()

    h3_ids = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(lat_flat, lon_flat)
    ]

    lookup = pd.DataFrame(
        {
            "pixel": np.arange(len(h3_ids), dtype=np.int64),
            "h3": h3_ids,
        }
    )

    out_file = lookup_dir() / f"{dataset_name}_lookup.parquet"

    lookup.to_parquet(out_file, index=False)

    print(f"Lookup saved: {out_file}")
    print(f"Pixels processed: {len(lookup)}")


def main():

    datasets = cfg["datasets"].keys()

    for dataset in datasets:
        build_lookup(dataset)


if __name__ == "__main__":
    main()