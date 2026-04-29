"""Plot BBAL residence index heatmap."""

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIES_NAME = "SAFS"
YEARS = (2022, 2023)
MIN_DAYS = 2


def load_species(year: int) -> pd.DataFrame:
    """Load species presence table for one year."""
    path = (
        paths["data"]
        / "features"
        / "species_presence"
        / f"year={year}"
        / "part.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def main() -> int:
    """Plot residence index heatmap."""
    bbox = cfg["region"]["bbox"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    bbox_poly = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
    coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])

    frames = []

    for year in YEARS:
        df_year = load_species(year)
        if df_year.empty:
            continue

        df_year = df_year[df_year["species"] == SPECIES_NAME].copy()
        frames.append(df_year)

    if not frames:
        raise RuntimeError(f"No records found for species: {SPECIES_NAME}")

    df = pd.concat(frames, ignore_index=True)
    df["residence_index"] = df["presence_count"] / df["individual_count"]

    residence = (
        df.groupby("h3", as_index=False)
        .agg(
            residence_index=("residence_index", "mean"),
            days=("date", "nunique"),
            total_presence=("presence_count", "sum"),
            max_individuals=("individual_count", "max"),
        )
    )

    residence = residence[residence["days"] >= MIN_DAYS].copy()
    residence["ri_log"] = np.log1p(residence["residence_index"])

    grid = load_grid(uint64=True)
    plot_gdf = grid.merge(residence, on="h3", how="left")

    ax = land.plot(color="grey", edgecolor="none", figsize=(10, 10))
    ax.set_facecolor("#e5fbfa")

    coast.plot(ax=ax, color="darkgrey", linewidth=0.5)

    grid.plot(
        ax=ax,
        edgecolor="darkgrey",
        facecolor="none",
        linewidth=0.2,
    )

    plot_gdf.dropna(subset=["ri_log"]).plot(
        ax=ax,
        column="ri_log",
        cmap="YlOrRd",
        legend=True,
        edgecolor="none",
        linewidth=0,
    )

    bbox_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1)

    mid_lat = (ymin + ymax) / 2
    scale = math.cos(math.radians(mid_lat))
    margin = 0.5

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin * scale, ymax + margin * scale)

    ax.set_title(
        f"{SPECIES_NAME} Residence Index Heatmap "
        f"(log1p, min {MIN_DAYS} days)"
    )

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())