"""Layer 2C: Wind aggregation from ERA5 to H3 grid."""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

from riskscape.config import cfg, paths


RAW_WIND_DIR = Path(cfg["paths"]["raw"]) / "wind"
LOOKUP_PATH = Path(paths["data"]) / "lookups" / "wind_lookup.parquet"
LAYER2_DIR = Path(paths["data"]) / "layer2"


def year_range():
    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def monthly_zip(year, month):
    name = (
        "derived-era5-single-levels-daily-statistics_"
        f"{year}_{month:02d}.zip"
    )
    return RAW_WIND_DIR / name


def open_month(path):
    with zipfile.ZipFile(path) as z:
        u_name = [n for n in z.namelist() if "u_component" in n][0]
        v_name = [n for n in z.namelist() if "v_component" in n][0]

        with z.open(u_name) as f:
            ds_u = xr.open_dataset(f).load()

        with z.open(v_name) as f:
            ds_v = xr.open_dataset(f).load()

    return xr.merge([ds_u, ds_v], compat="override")


def compute_wind(ds):
    return np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)


def assign_pixel(df):
    lat_name = "latitude" if "latitude" in df.columns else "lat"
    lon_name = "longitude" if "longitude" in df.columns else "lon"

    lats = np.sort(df[lat_name].unique())
    lons = np.sort(df[lon_name].unique())

    n_lon = len(lons)

    lat_to_idx = {v: i for i, v in enumerate(lats)}
    lon_to_idx = {v: i for i, v in enumerate(lons)}

    df["i_lat"] = df[lat_name].map(lat_to_idx)
    df["i_lon"] = df[lon_name].map(lon_to_idx)

    df["pixel"] = df["i_lat"] * n_lon + df["i_lon"]

    return df.drop(columns=["i_lat", "i_lon"])


def compute_anomaly(df, col):
    return df[col] - df.groupby("h3")[col].transform("mean")


def process_month(year, month, lookup):

    path = monthly_zip(year, month)

    if not path.exists():
        print(f"  Missing {year}-{month:02d}")
        return None

    print(f"  Processing {year}-{month:02d}")

    ds = open_month(path)

    u = ds["u10"]
    v = ds["v10"]

    df = (
        xr.Dataset({"u": u, "v": v})
        .to_dataframe()
        .reset_index()
    )

    ds.close()

    if "valid_time" in df.columns:
        df.rename(columns={"valid_time": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize() + pd.Timedelta(hours=9)

    df = assign_pixel(df)

    df = df.merge(lookup, on="pixel", how="left")
    df = df[df["h3"].notna()].copy()

    df["h3"] = df["h3"].apply(lambda x: int(x, 16)).astype("uint64")

    # aggregate u and v
    df = (
        df.groupby(["date", "h3"], as_index=False)
        .agg({"u": "mean", "v": "mean"})
    )

    # wind variables
    df["wind"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2).astype("float32")
    df["wind_u"] = df["u"].astype("float32")
    df["wind_v"] = df["v"].astype("float32")

    df["wind_dir"] = (
        np.degrees(np.arctan2(df["wind_v"], df["wind_u"])) % 360
    ).astype("float32")

    df = df.drop(columns=["u", "v"])

    return df


def process_year(year):

    print(f"\nProcessing wind for {year}")

    out_path = LAYER2_DIR / f"year={year}.parquet"

    if not out_path.exists():
        print("  Missing Layer 2 base file")
        return

    base = pd.read_parquet(out_path)

    drop_cols = [c for c in ["wind", "wind_u", "wind_v", "wind_dir", "wind_anom"] if c in base.columns]
    if drop_cols:
        print(f"  Dropping existing columns: {drop_cols}")
        base = base.drop(columns=drop_cols)

    lookup = pd.read_parquet(LOOKUP_PATH)

    monthly = []

    for month in range(1, 13):
        df = process_month(year, month, lookup)
        if df is not None:
            monthly.append(df)

    if not monthly:
        print("  No data")
        return

    wind_df = pd.concat(monthly, ignore_index=True)

    # anomaly after full year
    wind_df["wind_anom"] = compute_anomaly(wind_df, "wind").astype("float32")

    merged = base.merge(
        wind_df,
        on=["date", "h3"],
        how="left"
    )

    merged.to_parquet(out_path, index=False)

    print("  Saved:", out_path)
    print("  Rows:", len(merged))


def main():
    for year in year_range():
        process_year(year)


if __name__ == "__main__":
    main()