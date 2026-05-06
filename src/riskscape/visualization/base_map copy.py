"""Shared map helpers."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_AREA_COLOR = "#d62728"
STUDY_AREA_LINEWIDTH = 1.0


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


def label_colorbar_extremes(
    fig,
    bottom: str = "Low",
    top: str = "High",
) -> None:
    """Label a colorbar with simple endpoint text."""
    if len(fig.axes) < 2:
        return

    cax = fig.axes[-1]
    cax.text(
        0.5,
        -0.03,
        bottom,
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    cax.text(
        0.5,
        1.03,
        top,
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_compact_colorbar(
    ax,
    cmap: str,
    norm: colors.Normalize,
    label: str,
    bottom_label: str = "Low",
    top_label: str = "High",
    invert: bool = False,
) -> None:
    """Draw a compact colorbar matching feature-map legends."""
    fig = ax.figure
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
        ax=ax,
        label=label,
        ticks=[],
        shrink=0.72,
        pad=0.02,
        fraction=0.035,
    )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(which="both", length=0)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("face")

    if invert:
        cbar.ax.invert_yaxis()

    label_colorbar_extremes(fig, bottom=bottom_label, top=top_label)


def format_depth_label(value: float) -> str:
    """Return compact depth label text."""
    return f"{value:,.0f} m"


def bathymetry_norm(
    values: pd.Series,
    vmax_quantile: float,
) -> colors.Normalize:
    """Return bathymetry color normalization."""
    vmin = 0.0
    vmax = values.quantile(vmax_quantile)

    if vmax <= vmin:
        vmax = values.max()

    return colors.Normalize(vmin=vmin, vmax=vmax)


def draw_bathymetry_base_layer(
    ax,
    cmap: str = "Blues",
    legend: bool = True,
    vmax_quantile: float = 1,
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

    values = plot_gdf[column].dropna()
    norm = bathymetry_norm(values, vmax_quantile)

    plot_gdf.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=False,
        edgecolor="none",
        linewidth=0,
        vmin=norm.vmin,
        vmax=norm.vmax,
    )

    if legend:
        draw_compact_colorbar(
            ax,
            cmap,
            norm,
            label="Depth",
            bottom_label=format_depth_label(norm.vmax),
            top_label=format_depth_label(norm.vmin),
            invert=True,
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
    draw_study_area_boundary(ax, bbox_gdf)


def draw_study_area_boundary(
    ax,
    bbox_gdf: gpd.GeoDataFrame,
    linewidth: float = STUDY_AREA_LINEWIDTH,
) -> None:
    """Draw the configured study-area boundary."""
    bbox_gdf.boundary.plot(
        ax=ax,
        edgecolor=STUDY_AREA_COLOR,
        linewidth=linewidth,
    )


def format_coordinate_axes(ax) -> None:
    """Show longitude and latitude axes on map figures."""
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=9)
    ax.grid(
        True,
        color="white",
        linewidth=0.6,
        alpha=0.45,
    )


def draw_north_arrow(ax) -> None:
    """Draw a simple north arrow in axes coordinates."""
    ax.annotate(
        "N",
        xy=(0.92, 0.93),
        xytext=(0.92, 0.83),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        color="#f6f6f6",
        fontsize=10,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#f6f6f6",
            "linewidth": 0.9,
            "mutation_scale": 10,
            "path_effects": [
                path_effects.withStroke(linewidth=1.2, foreground="#666666")
            ],
        },
        zorder=10,
        path_effects=[
            path_effects.withStroke(linewidth=1.2, foreground="#666666")
        ],
    )


def draw_reference_inset(
    ax,
    land: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
) -> None:
    """Draw a small global reference map with the region bbox."""
    inset_ax = ax.inset_axes([0.73, 0.025, 0.22, 0.18])
    inset_ax.set_facecolor((1, 1, 1, 0.72))

    land.plot(
        ax=inset_ax,
        color="#b8b8b8",
        edgecolor="none",
        linewidth=0,
    )
    draw_study_area_boundary(inset_ax, bbox_gdf, linewidth=1.2)

    inset_ax.set_xlim(-180, 180)
    inset_ax.set_ylim(-90, 90)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#9a9a9a")
        spine.set_linewidth(0.6)


def draw_map_context(
    ax,
    bbox_gdf: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
    show_north_arrow: bool = True,
    show_reference_map: bool = True,
) -> None:
    """Draw reference overlays and optional map decorations."""
    draw_reference_layers(ax, bbox_gdf, land, coast)

    if show_north_arrow:
        draw_north_arrow(ax)

    if show_reference_map:
        draw_reference_inset(ax, land, bbox_gdf)


def format_map_axes(
    ax,
    title: str,
    show_coordinates: bool = True,
) -> None:
    """Apply title and coordinate styling."""
    ax.set_title(title)

    if show_coordinates:
        format_coordinate_axes(ax)
    else:
        ax.set_axis_off()
