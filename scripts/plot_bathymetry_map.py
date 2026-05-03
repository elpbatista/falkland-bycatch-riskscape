"""Plot H3 bathymetry map with land on top."""

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
import numpy as np

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Plot bathymetry from static H3 features."""
    bbox = cfg["region"]["bbox"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    bbox_poly = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
    coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])

    static_path = (
        paths["data"]
        / "features"
        / "static"
        / "static.parquet"
    )

    if not static_path.exists():
        raise FileNotFoundError(f"Static features not found: {static_path}")

    static = pd.read_parquet(static_path)

    grid = load_grid(uint64=True)
    plot_gdf = grid.merge(static, on="h3", how="left")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#e5fbfa")

    # Bathymetry (base layer)
    plot_gdf["depth_log"] = np.log1p(plot_gdf["depth_m"])
    # plot_gdf["dist_coast_km"] = plot_gdf["dist_coast_m"] / 1000.0
    # plot_gdf["dist_coast_km_log"] = np.log1p(plot_gdf["dist_coast_km"])

    plot_gdf.plot(
        ax=ax,
        column="depth_log",
        # column="slope",
        # column = "dist_coast_km_log",
        cmap="Blues",
        # cmap="inferno",
        # cmap="viridis",
        legend=True,
        edgecolor="none",
        linewidth=0,
        vmin=0,
        vmax=plot_gdf["depth_log"].quantile(0.98),
    )

    # Grid (optional)
    grid.plot(
        ax=ax,
        edgecolor="darkgrey",
        facecolor="none",
        linewidth=0.1,
    )

    # Coastline
    coast.plot(ax=ax, color="darkgrey", linewidth=0.5)

    # Land (on top)
    land.plot(ax=ax, color="grey", edgecolor="none")

    # Bounding box
    bbox_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1)

    mid_lat = (ymin + ymax) / 2
    scale = math.cos(math.radians(mid_lat))
    margin = 0.5

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin * scale, ymax + margin * scale)

    ax.set_title("Bathymetry Depth Map (H3 mean depth, m)")
    # ax.set_title("Bathymetry Slope Map (H3 mean slope, m/m)")
    # ax.set_title("Distance to Coast Map (H3 mean distance, km)")

    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())