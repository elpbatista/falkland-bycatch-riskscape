"""Plot mean wind variables over the H3 grid with Falkland Islands boundary."""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from riskscape.config import cfg, paths


def main():

    layer2_dir = paths.get("layer2", paths["data"] / "layer2")
    plots_dir = paths.get("plots", paths["plots"])

    plots_dir.mkdir(parents=True, exist_ok=True)

    resolution = cfg["grid"]["resolution"]
    region = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region}.geojson"
    falklands_file = paths.get("falklands")

    print("Loading grid")
    grid = gpd.read_file(grid_file)

    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    print("Loading Falkland Islands boundary")
    if falklands_file:
        falklands = gpd.read_file(falklands_file)
        if falklands.crs != "EPSG:4326":
            falklands = falklands.to_crs("EPSG:4326")
        print(f"  Loaded: {falklands.crs}")
    else:
        print("  Warning: Falklands geometry not found in config")
        falklands = None

    parquet_files = sorted(layer2_dir.glob("year=*.parquet"))
    years = [int(f.stem.split("=")[1]) for f in parquet_files]

    print(f"Found {len(years)} years: {years}")

    for year in years:
        print(f"\nProcessing {year}")

        print("  Loading layer2 data")
        df = pd.read_parquet(layer2_dir / f"year={year}.parquet")

        # -------- WIND --------
        print("  Computing mean wind")
        mean_wind = df.groupby("h3")["wind"].mean().reset_index()

        # -------- WIND GRAD --------
        print("  Computing mean wind_grad")
        mean_wind_grad = df.groupby("h3")["wind_grad"].mean().reset_index()

        # -------- WIND ANOM --------
        print("  Computing mean wind_anom")
        mean_wind_anom = df.groupby("h3")["wind_anom"].mean().reset_index()

        # Join separately (same pattern as your SST script)
        grid_wind = grid.merge(mean_wind, on="h3", how="left")
        grid_grad = grid.merge(mean_wind_grad, on="h3", how="left")
        grid_anom = grid.merge(mean_wind_anom, on="h3", how="left")

        # ---------------- PLOTTING ----------------

        def plot_map(gdf, column, title, cmap, vmin=None, vmax=None, fname="plot.png"):

            fig, ax = plt.subplots(figsize=(12, 10))

            gdf.plot(
                column=column,
                cmap=cmap,
                linewidth=0,
                legend=True,
                ax=ax,
                vmin=vmin,
                vmax=vmax
            )

            if falklands is not None:
                falklands.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.5,
                    label="Falkland Islands"
                )

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(loc="upper right")

            plt.tight_layout()

            output_file = plots_dir / fname
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"  Saved to {output_file}")

            plt.close()

        # Wind speed
        plot_map(
            grid_wind,
            "wind",
            f"Mean Wind Speed ({year}) [m/s]",
            cmap="viridis",
            vmin=0,
            vmax=15,
            fname=f"mean_wind_{year}.png"
        )

        # Wind gradient
        plot_map(
            grid_grad,
            "wind_grad",
            f"Mean Wind Gradient ({year})",
            cmap="magma",
            fname=f"mean_wind_grad_{year}.png"
        )

        # Wind anomaly
        plot_map(
            grid_anom,
            "wind_anom",
            f"Mean Wind Anomaly ({year})",
            cmap="RdBu_r",
            fname=f"mean_wind_anom_{year}.png"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()