"""
Plot 10-year mean SST gradient on H3 grid.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from riskscape.config import cfg, paths


def load_all_years():
    """Load and concatenate all Layer 2 yearly files."""

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    frames = []

    for year in range(start_year, end_year + 1):
        file_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"
        df = pd.read_parquet(file_path, columns=["h3", "sst_grad"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():

    print("Loading Layer 2 data")
    df = load_all_years()

    print("Computing 10-year mean SST gradient")
    mean_grad = (
        df.groupby("h3", as_index=False)["sst_grad"]
        .mean()
    )

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    print("Loading H3 grid")
    grid = gpd.read_parquet(grid_path)

    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    print("Merging mean gradient")
    grid = grid.merge(mean_grad, on="h3", how="left")

    print("Plotting")
    fig, ax = plt.subplots(figsize=(10, 8))

    grid.plot(
        column="sst_grad",
        ax=ax,
        legend=True
    )

    ax.set_title("10-Year Mean SST Gradient")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()