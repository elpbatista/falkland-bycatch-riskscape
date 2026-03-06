"""Plot mean SST over the H3 grid."""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import cfg, paths


def main():

    layer1_dir = paths.get("layer1", paths["data"] / "layer1")

    resolution = cfg["grid"]["resolution"]
    region = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region}.geojson"

    print("Loading grid")
    grid = gpd.read_file(grid_file)

    # Convert H3 index from hex string → uint64
    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    print("Loading layer1 data")
    df = pd.read_parquet(layer1_dir / "year=2023.parquet")

    print("Computing mean SST")
    mean_sst = (
        df.groupby("h3")["sst"]
        .mean()
        .reset_index()
    )

    mean_sst["sst"] = mean_sst["sst"] - 273.15

    print("Joining grid + SST")
    grid = grid.merge(mean_sst, on="h3", how="left")

    print("Plotting")

    fig, ax = plt.subplots(figsize=(10, 8))

    grid.plot(
        column="sst",
        cmap="coolwarm",
        linewidth=0,
        legend=True,
        ax=ax
    )

    ax.set_title("Mean SST (2023)")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()