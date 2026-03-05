"""Validate raster–H3 extraction setup before full run."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from riskscape.config import cfg, paths


TEST_DAY = "2014-01-01"


def detect_coords(ds):

    lat = None
    lon = None

    for name in ("lat", "latitude"):
        if name in ds.coords:
            lat = name
            break

    for name in ("lon", "longitude"):
        if name in ds.coords:
            lon = name
            break

    if lat is None or lon is None:
        raise RuntimeError("Could not detect lat/lon coordinates")

    return lat, lon


def load_lookup(dataset_name):

    lookup_dir = paths.get("lookups", paths["data"] / "lookups")
    lookup_path = lookup_dir / f"{dataset_name}_lookup.parquet"

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup not found: {lookup_path}")

    return pd.read_parquet(lookup_path)


def open_dataset(dataset_name):

    dataset_dir = paths["raw"] / dataset_name
    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        raise RuntimeError(f"No files found for {dataset_name}")

    return xr.open_mfdataset(files, combine="by_coords")


def select_day(dataset_name, day):

    dataset_dir = paths["raw"] / dataset_name
    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        raise RuntimeError(f"No files found for {dataset_name}")

    ds = xr.open_dataset(files[0])

    if "time" not in ds.coords:
        return ds

    target = np.datetime64(day)

    try:
        subset = ds.sel(time=target, method="nearest")
    except Exception:
        raise RuntimeError(f"No data for {day}")

    return subset


def main():

    print("\nVALIDATION TEST\n")

    expected_h3 = None

    for dataset_name in cfg["layer1"]["variables"]:

        print(f"\nDataset: {dataset_name}")

        var = cfg["datasets"][dataset_name]["variable"]

        ds = open_dataset(dataset_name)

        lat_name, lon_name = detect_coords(ds)

        print("coords:", lat_name, lon_name)

        lookup = load_lookup(dataset_name)

        lookup_rows = len(lookup)
        unique_h3 = lookup["h3"].nunique()

        print("lookup rows:", lookup_rows)
        print("unique H3 cells:", unique_h3)

        if expected_h3 is None:
            expected_h3 = unique_h3

        ds_day = select_day(dataset_name, TEST_DAY)

        if var not in ds_day.variables:
            raise RuntimeError(f"Variable '{var}' not found")

        da = ds_day[var]

        print("variable:", var)
        print("units:", da.attrs.get("units", "unknown"))

        values = da.values.reshape(-1)

        raster_pixels = len(values)

        print("raster pixels:", raster_pixels)

        valid_pixels = np.isfinite(values).sum()

        print("valid pixels:", valid_pixels)

        if valid_pixels == 0:
            raise RuntimeError("Raster contains no valid data")

        ds.close()

    print("\nExpected H3 cells:", expected_h3)

    if expected_h3 < 10000:
        print("WARNING: H3 coverage seems too small")

    print("\nValidation completed successfully.\n")


if __name__ == "__main__":
    main()