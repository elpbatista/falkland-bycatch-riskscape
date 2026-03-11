"""
Compute and plot Chlorophyll front frequency (10-year).
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from riskscape.config import cfg, paths


def load_all_years():
    """Load chlorophyll gradients."""

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    frames = []

    for year in range(start_year, end_year + 1):
        file_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"
        df = pd.read_parquet(file_path, columns=["h3", "chl_grad"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():

    print("Loading chlorophyll gradients")
    df = load_all_years()

    print("Computing threshold (75th percentile)")
    tau = df["chl_grad"].quantile(0.75)
    print("Threshold τ_chl =", tau)

    df["front_flag"] = (df["chl_grad"] > tau).astype("int8")

    freq = (
        df.groupby("h3", as_index=False)["front_flag"]
        .mean()
        .rename(columns={"front_flag": "chl_front_frequency"})
    )

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    print("Loading grid")
    grid = gpd.read_parquet(grid_path)
    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    print("Merging")
    grid = grid.merge(freq, on="h3", how="left")

    print("Plotting")
    fig, ax = plt.subplots(figsize=(10, 8))

    grid.plot(
        column="chl_front_frequency",
        ax=ax,
        legend=True
    )

    ax.set_title("Chlorophyll Front Frequency (Top 25% Gradient)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()