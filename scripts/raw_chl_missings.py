"""Check CHL missingness across all days (raw data)."""

import numpy as np
import pandas as pd
import xarray as xr

FILE = (
    "data/raw/chl/"
    "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_CHL_64.73W-50.27W_57.44S-46.56S_2014-01-01-2023-12-31.nc"
)


def main():

    print("Opening dataset")
    ds = xr.open_dataset(FILE)

    var_name = list(ds.data_vars)[0]
    print(f"Using variable: {var_name}")

    da = ds[var_name]

    times = ds["time"].values

    results = []

    print("Processing time series")

    for t in times:
        data = da.sel(time=t).values

        total = data.size
        nan_count = np.isnan(data).sum()

        results.append({
            "date": pd.to_datetime(t),
            "nan_count": nan_count,
            "nan_fraction": nan_count / total,
        })

    df = pd.DataFrame(results)

    print("\n=== SUMMARY ===")
    print(df["nan_fraction"].describe())

    print("\nDays with NO NaNs:")
    print((df["nan_fraction"] == 0).sum())

    print("\nDays with NaNs:")
    print((df["nan_fraction"] > 0).sum())

    print("\nMax missing fraction:")
    print(df["nan_fraction"].max())

    print("\nMin missing fraction:")
    print(df["nan_fraction"].min())

    # --- Save for inspection ---
    df.to_csv("data/chl_missing_timeseries.csv", index=False)
    print("\nSaved: data/chl_missing_timeseries.csv")


if __name__ == "__main__":
    main()