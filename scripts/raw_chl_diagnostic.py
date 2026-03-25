"""Strict diagnostic of RAW CHL (no assumptions, no plotting tricks)."""

import numpy as np
import xarray as xr

FILE = (
    "data/raw/chl/"
    "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_CHL_64.73W-50.27W_57.44S-46.56S_2014-01-01-2023-12-31.nc"
)


def main():

    print("Opening dataset")
    ds = xr.open_dataset(FILE)

    print("\nVariables:")
    print(list(ds.data_vars))

    print("\nDimensions:")
    print(ds.dims)

    # --- REQUIRE USER CONFIRMATION ---
    var_name = list(ds.data_vars)[0]
    print(f"\nUsing variable: {var_name}")

    da = ds[var_name]

    # --- EXACT DATE ---
    date = np.datetime64("2022-07-02")

    print(f"\nSelecting date: {date}")

    da_day = da.sel(time=date)

    data = da_day.values

    print("\n=== RAW CHECK ===")
    print("Shape:", data.shape)
    print("Total cells:", data.size)

    nan_count = np.isnan(data).sum()
    finite_count = np.isfinite(data).sum()

    print("NaN count:", nan_count)
    print("Finite count:", finite_count)

    print("Min (finite):", np.nanmin(data))
    print("Max (finite):", np.nanmax(data))

    # --- CRITICAL CHECK ---
    if nan_count == 0:
        print("\nRESULT: NO missing data in RAW CHL")
    else:
        print("\nRESULT: RAW CHL contains missing data")


if __name__ == "__main__":
    main()