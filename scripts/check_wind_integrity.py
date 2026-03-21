"""Integrity check for ERA5 daily wind downloads."""

from pathlib import Path

import xarray as xr
import numpy as np


WIND_DIR = Path("data/raw/wind")


def expected_files():
    """Return expected monthly filenames."""
    files = []
    for year in range(2014, 2024):
        for month in range(1, 13):
            name = (
                f"derived-era5-single-levels-daily-statistics_"
                f"{year}_{month:02d}.nc"
            )
            files.append(name)
    return files


def main():

    print("\nChecking ERA5 daily wind integrity\n")

    missing = []
    unreadable = []

    for fname in expected_files():

        path = WIND_DIR / fname

        if not path.exists():
            missing.append(fname)
            continue

        try:
            ds = xr.open_dataset(path)

            # Basic structure checks
            if "u10" not in ds.variables or "v10" not in ds.variables:
                print("Missing variables in:", fname)

            if "time" not in ds.coords:
                print("Missing time dimension in:", fname)

            # Check data is not fully NaN
            u_vals = ds["u10"].values
            v_vals = ds["v10"].values

            if np.isnan(u_vals).all():
                print("All NaN in u10:", fname)

            if np.isnan(v_vals).all():
                print("All NaN in v10:", fname)

            ds.close()

        except Exception:
            unreadable.append(fname)

    print("Total expected files:", len(expected_files()))
    print("Missing files:", len(missing))
    print("Unreadable files:", len(unreadable))

    if missing:
        print("\nMissing:")
        for m in missing:
            print(" -", m)

    if unreadable:
        print("\nUnreadable:")
        for u in unreadable:
            print(" -", u)

    print("\nIntegrity check completed.\n")


if __name__ == "__main__":
    main()