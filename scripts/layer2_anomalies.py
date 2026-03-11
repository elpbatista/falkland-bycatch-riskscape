"""
Compute and merge Layer 2 anomalies (sst_anom, chl_anom, ssh_anom)
into existing layer2/year=YYYY.parquet files.

Climatology method: daily DOY per H3 cell.
"""

from pathlib import Path

import pandas as pd

from riskscape.config import cfg, paths


def load_full_layer1():
    """Load full 10-year Layer 1 dataset."""
    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    frames = []

    for year in range(start_year, end_year + 1):
        file_path = Path(paths["layer1"]) / f"year={year}.parquet"
        df = pd.read_parquet(file_path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_climatology(df):
    """Compute DOY climatology per H3 cell."""
    df["date"] = pd.to_datetime(df["date"])
    df["doy"] = df["date"].dt.dayofyear

    climatology = (
        df.groupby(["h3", "doy"], as_index=False)
        .agg(
            sst_clim=("sst", "mean"),
            chl_clim=("chl", "mean"),
            ssh_clim=("ssh", "mean"),
        )
    )

    return climatology


def process_year(year, climatology):
    """Compute anomalies and merge into existing Layer 2 table."""
    print(f"Processing anomalies for {year}")

    layer1_path = Path(paths["layer1"]) / f"year={year}.parquet"
    layer2_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"

    df1 = pd.read_parquet(layer1_path)
    df1["date"] = pd.to_datetime(df1["date"])
    df1["doy"] = df1["date"].dt.dayofyear

    df1 = df1.merge(climatology, on=["h3", "doy"], how="left")

    df_anom = df1[["date", "h3"]].copy()
    df_anom["sst_anom"] = (df1["sst"] - df1["sst_clim"]).astype("float32")
    df_anom["chl_anom"] = (df1["chl"] - df1["chl_clim"]).astype("float32")
    df_anom["ssh_anom"] = (df1["ssh"] - df1["ssh_clim"]).astype("float32")

    df2 = pd.read_parquet(layer2_path)

    df2 = df2.merge(df_anom, on=["date", "h3"], how="left")

    df2.to_parquet(layer2_path, index=False)

    print("Updated:", layer2_path)
    print("Rows:", len(df2))


def main():

    print("Loading full Layer 1 dataset")
    df_full = load_full_layer1()

    print("Computing climatology")
    climatology = compute_climatology(df_full)

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    for year in range(start_year, end_year + 1):
        process_year(year, climatology)


if __name__ == "__main__":
    main()