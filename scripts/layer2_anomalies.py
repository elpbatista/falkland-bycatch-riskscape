"""Compute and merge Layer 2 anomalies into existing yearly tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import cfg, paths


def year_range():
    """Return inclusive year range from config."""

    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def safe_log_chl(series):
    """Return log10(CHL), masking non-positive values."""

    values = series.astype("float32").copy()
    values[values <= 0] = np.nan
    return np.log10(values)


def load_full_layer1():
    """Load full Layer 1 dataset."""

    frames = []

    for year in year_range():
        file_path = Path(paths["layer1"]) / f"year={year}.parquet"
        df = pd.read_parquet(file_path, columns=["date", "h3", "sst", "chl", "ssh"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_climatology(df):
    """Compute DOY climatology per H3 cell."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear
    df["chl_log"] = safe_log_chl(df["chl"])

    climatology = (
        df.groupby(["h3", "doy"], as_index=False)
        .agg(
            sst_clim=("sst", "mean"),
            chl_log_clim=("chl_log", "mean"),
            ssh_clim=("ssh", "mean"),
        )
    )

    return climatology


def process_year(year, climatology):
    """Compute anomalies and merge into existing Layer 2 table."""

    print(f"Processing anomalies for {year}")

    layer1_path = Path(paths["layer1"]) / f"year={year}.parquet"
    layer2_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"

    df1 = pd.read_parquet(layer1_path, columns=["date", "h3", "sst", "chl", "ssh"])
    df1["date"] = pd.to_datetime(df1["date"])
    df1["doy"] = df1["date"].dt.dayofyear
    df1["chl_log"] = safe_log_chl(df1["chl"])

    df1 = df1.merge(climatology, on=["h3", "doy"], how="left")

    df_anom = df1[["date", "h3"]].copy()
    df_anom["sst_anom"] = (df1["sst"] - df1["sst_clim"]).astype("float32")
    df_anom["chl_anom"] = (
        df1["chl_log"] - df1["chl_log_clim"]
    ).astype("float32")
    df_anom["ssh_anom"] = (df1["ssh"] - df1["ssh_clim"]).astype("float32")

    df2 = pd.read_parquet(layer2_path)

    drop_cols = [c for c in ["sst_anom", "chl_anom", "ssh_anom"] if c in df2.columns]
    if drop_cols:
        print(f"Dropping existing columns: {drop_cols}")
        df2 = df2.drop(columns=drop_cols)

    df2 = df2.merge(df_anom, on=["date", "h3"], how="left")

    df2.to_parquet(layer2_path, index=False)

    print("Updated:", layer2_path)
    print("Rows:", len(df2))


def main():
    """Run anomaly computation for all years."""

    print("Loading full Layer 1 dataset")
    df_full = load_full_layer1()

    print("Computing climatology")
    climatology = compute_climatology(df_full)

    for year in year_range():
        process_year(year, climatology)


if __name__ == "__main__":
    main()