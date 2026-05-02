"""Plot mean SST over the H3 grid for all years with Falkland Islands boundary."""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from riskscape.config import cfg, paths


def main():

    layer1_dir = paths.get("layer1", paths["data"] / "layer1")
    plots_dir = paths.get("plots", paths["plots"])

    # Create plots directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    resolution = cfg["grid"]["resolution"]
    region = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region}.geojson"
    falklands_file = paths.get("falklands")

    print("Loading grid")
    grid = gpd.read_file(grid_file)

    # Convert H3 index from hex string → uint64
    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    # Load Falkland Islands boundary
    print("Loading Falkland Islands boundary")
    if falklands_file:
        falklands = gpd.read_file(falklands_file)
        # Reproject to WGS84 if needed
        if falklands.crs != "EPSG:4326":
            falklands = falklands.to_crs("EPSG:4326")
        print(f"  Loaded: {falklands.crs}")
    else:
        print("  Warning: Falklands geometry not found in config")
        falklands = None

    # Find all parquet files and extract years
    parquet_files = sorted(layer1_dir.glob("year=*.parquet"))
    years = [int(f.stem.split("=")[1]) for f in parquet_files]

    print(f"Found {len(years)} years: {years}")

    for year in years:
        print(f"\nProcessing {year}")
        
        print(f"  Loading layer1 data")
        df = pd.read_parquet(layer1_dir / f"year={year}.parquet")

        print(f"  Computing mean SST")
        mean_sst = (
            df.groupby("h3")["sst"]
            .mean()
            .reset_index()
        )

        # Convert Kelvin to Celsius
        mean_sst["sst"] = mean_sst["sst"] - 273.15

        print(f"  Joining grid + SST")
        grid_year = grid.merge(mean_sst, on="h3", how="left")

        print(f"  Plotting and saving")

        fig, ax = plt.subplots(figsize=(12, 10))

        grid_year.plot(
            column="sst",
            cmap="coolwarm",
            linewidth=0,
            legend=True,
            ax=ax,
            vmin=0,
            vmax=15
        )

        # Overlay Falkland Islands boundary
        if falklands is not None:
            falklands.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                label="Falkland Islands"
            )

        ax.set_title(f"Mean SST ({year})\nTemperature in °C", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")

        plt.tight_layout()
        
        output_file = plots_dir / f"mean_sst_{year}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_file}")
        
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()