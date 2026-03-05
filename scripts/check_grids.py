"""Check grid stability for all raster datasets."""

from pathlib import Path

import numpy as np
import xarray as xr


RAW_DIR = Path("data/raw")


def get_coords(ds):
    """Return latitude and longitude coordinate names."""

    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude"]

    lat_name = None
    lon_name = None

    for c in lat_candidates:
        if c in ds.coords:
            lat_name = c
            break

    for c in lon_candidates:
        if c in ds.coords:
            lon_name = c
            break

    if lat_name is None or lon_name is None:
        raise ValueError("Could not find lat/lon coordinates")

    return lat_name, lon_name


def check_dataset(dataset_dir):
    """Verify that all rasters share the same grid."""

    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        print(f"{dataset_dir.name}: no files found")
        return

    print(f"\nChecking dataset: {dataset_dir.name}")

    ds0 = xr.open_dataset(files[0])

    lat_name, lon_name = get_coords(ds0)

    lat0 = ds0[lat_name].values
    lon0 = ds0[lon_name].values

    var0 = list(ds0.data_vars)[0]
    shape0 = ds0[var0].shape

    ds0.close()

    for f in files[1:]:

        ds = xr.open_dataset(f)

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        var = list(ds.data_vars)[0]
        shape = ds[var].shape

        lat_ok = np.array_equal(lat, lat0)
        lon_ok = np.array_equal(lon, lon0)
        shape_ok = shape == shape0

        if not (lat_ok and lon_ok and shape_ok):
            print(f"Grid mismatch detected in {f.name}")
            ds.close()
            return

        ds.close()

    print("Grid is stable")


def main():

    datasets = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

    for dataset_dir in datasets:
        check_dataset(dataset_dir)


if __name__ == "__main__":
    main()