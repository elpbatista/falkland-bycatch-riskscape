"""Validate Layer 1b dynamic seascapes (full grid + explicit date)."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import cfg, paths


def load_grid():
    resolution = cfg["grid"]["resolution"]
    region = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region}.geojson"

    grid = gpd.read_file(grid_file)
    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    return grid


def load_falklands():
    falklands_file = paths.get("falklands")

    if not falklands_file:
        print("Warning: Falklands boundary not configured")
        return None

    gdf = gpd.read_file(falklands_file)

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def class_distribution(df):
    print("\nClass distribution:")
    print(df["regime_id"].value_counts(dropna=False))


def temporal_stability(df):
    stats = df.groupby("date")["regime_id"].nunique()

    print("\nTemporal stability (unique classes per day):")
    print(stats.describe())


def plot_mean_regime(df, grid, falklands, year, plots_dir):
    print("  Computing dominant regime")

    mode_df = (
        df.dropna(subset=["regime_id"])
        .groupby("h3")["regime_id"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
        .reset_index()
    )

    grid_year = grid.merge(mode_df, on="h3", how="left")

    fig, ax = plt.subplots(figsize=(12, 10))

    grid_year.plot(
        column="regime_id",
        cmap="tab10",
        linewidth=0,
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "No data",
        },
    )

    if falklands is not None:
        falklands.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
        )

    ax.set_title(f"Dominant Regime ({year})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()

    out_file = plots_dir / f"regime_mode_{year}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {out_file}")


def plot_single_day(df, grid, falklands, year, plots_dir):
    """Plot regime map for a specific day (full grid)."""

    # Pick a reproducible sample date (middle of year)
    unique_dates = sorted(pd.to_datetime(df["date"].unique()))
    sample_date = unique_dates[len(unique_dates) // 2]

    print(f"  Plotting sample day: {sample_date}")

    day_df = df[df["date"] == sample_date]

    grid_day = grid.merge(day_df, on="h3", how="left")

    fig, ax = plt.subplots(figsize=(12, 10))

    grid_day.plot(
        column="regime_id",
        cmap="tab10",
        linewidth=0,
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "label": "No data",
        },
    )

    if falklands is not None:
        falklands.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
        )

    ax.set_title(
        f"Regimes on {sample_date.strftime('%Y-%m-%d')}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()

    out_file = plots_dir / f"regime_day_{year}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {out_file}")


def main():

    layer1b_dir = Path(paths["data"]) / "layer1b"
    plots_dir = paths.get("plots", paths["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid()
    falklands = load_falklands()

    files = sorted(layer1b_dir.glob("year=*.parquet"))
    years = [int(f.stem.split("=")[1]) for f in files]

    print(f"Found years: {years}")

    for year in years:
        print(f"\nProcessing {year}")

        df = pd.read_parquet(layer1b_dir / f"year={year}.parquet")

        class_distribution(df)
        temporal_stability(df)

        plot_mean_regime(df, grid, falklands, year, plots_dir)
        plot_single_day(df, grid, falklands, year, plots_dir)

    print("\nDone")


if __name__ == "__main__":
    main()