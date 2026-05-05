"""Shared map helpers."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box
import numpy as np

from riskscape.config import cfg, paths
from riskscape.grid import load_grid



PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_reference_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load land and coastline layers."""
    land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
    coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])

    return land, coast


def load_static_features() -> pd.DataFrame:
    """Load static H3 features."""
    static_path = (
        paths["data"]
        / "features"
        / "static"
        / "static.parquet"
    )

    if not static_path.exists():
        raise FileNotFoundError(f"Static features not found: {static_path}")

    return pd.read_parquet(static_path)


def setup_map(figsize: tuple[int, int] = (10, 10)):
    """Create base map axis and bbox layer."""
    bbox = cfg["region"]["bbox"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    bbox_poly = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#e5fbfa")

    mid_lat = (ymin + ymax) / 2
    scale = math.cos(math.radians(mid_lat))
    margin = 0.5

    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin * scale, ymax + margin * scale)

    return fig, ax, bbox_gdf


def draw_bathymetry_base_layer(
    ax,
    cmap: str = "Blues",
    legend: bool = True,
    vmax_quantile: float = 0.98,
    draw_grid: bool = True,
    log_scale: bool = False,
) -> gpd.GeoDataFrame:
    """Draw H3 bathymetry as a base layer."""
    static = load_static_features()
    grid = load_grid(uint64=True)
    plot_gdf = grid.merge(static, on="h3", how="left")

    column = "depth_m"

    if log_scale:
        column = "depth_log"
        depth = plot_gdf["depth_m"].clip(lower=0)
        plot_gdf[column] = np.log1p(depth)

    plot_gdf.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=legend,
        edgecolor="none",
        linewidth=0,
        vmin=0,
        vmax=plot_gdf[column].quantile(vmax_quantile),
    )

    if draw_grid:
        grid.plot(
            ax=ax,
            edgecolor="darkgrey",
            facecolor="none",
            linewidth=0.1,
        )

    return plot_gdf


def draw_reference_layers(
    ax,
    bbox_gdf: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw reference layers on top."""
    coast.plot(ax=ax, color="darkgrey", linewidth=0.5)
    land.plot(ax=ax, color="grey", edgecolor="none")
    bbox_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=1)
