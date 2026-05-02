"""Shared map helpers."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from riskscape.config import cfg


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_reference_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load land and coastline layers."""
    land = gpd.read_file(PROJECT_ROOT / cfg["references"]["land"])
    coast = gpd.read_file(PROJECT_ROOT / cfg["references"]["coastline"])

    return land, coast


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