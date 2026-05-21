"""Shared map helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import geopandas as gpd
from matplotlib import colors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from riskscape.config import cfg, paths
from riskscape.grid import load_grid
from riskscape.visualization.legends import (
    LegendStyle,
    draw_continuous_colorbar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_CRS = "EPSG:4326"
OCEAN_COLOR = "#e5fbfa"
LAND_COLOR = "grey"
COAST_COLOR = "darkgrey"
GRID_COLOR = "darkgrey"
INSET_LAND_COLOR = "#b8b8b8"
INSET_FACE_COLOR = (1, 1, 1, 0.72)
STUDY_AREA_COLOR = "#cfcfcf"
STUDY_AREA_LINEWIDTH = 1.0
COLORBAR_KWARGS = {
    "ticks": [],
    "shrink": 0.72,
    "pad": 0.02,
    "fraction": 0.035,
}


@dataclass(frozen=True)
class MapBounds:
    """Configured map extent."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_config(cls) -> MapBounds:
        """Build bounds from project config."""
        bbox = cfg["region"]["bbox"]
        return cls(
            xmin=bbox["xmin"],
            ymin=bbox["ymin"],
            xmax=bbox["xmax"],
            ymax=bbox["ymax"],
        )

    @property
    def mid_latitude(self) -> float:
        """Return the vertical midpoint latitude."""
        return (self.ymin + self.ymax) / 2

    @property
    def latitude_scale(self) -> float:
        """Return longitude-to-latitude margin scale for this extent."""
        return math.cos(math.radians(self.mid_latitude))

    def geometry(self):
        """Return the bounding geometry."""
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def apply_to_axis(self, ax, margin: float = 0.5) -> None:
        """Apply the map extent to an axis."""
        ax.set_xlim(self.xmin - margin, self.xmax + margin)
        ax.set_ylim(
            self.ymin - margin * self.latitude_scale,
            self.ymax + margin * self.latitude_scale,
        )


def project_reference_path(key: str) -> Path:
    """Return an absolute reference-layer path from config."""
    return PROJECT_ROOT / cfg["references"][key]


def load_reference_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load land and coastline layers."""
    land = gpd.read_file(project_reference_path("land"))
    coast = gpd.read_file(project_reference_path("coastline"))

    return land, coast


def static_feature_path() -> Path:
    """Return the static feature table path."""
    return paths["data"] / "features" / "static" / "static.parquet"


def load_static_features() -> pd.DataFrame:
    """Load static H3 features."""
    static_path = static_feature_path()

    if not static_path.exists():
        raise FileNotFoundError(f"Static features not found: {static_path}")

    return pd.read_parquet(static_path)


def setup_map(figsize: tuple[int, int] = (10, 10)):
    """Create base map axis and bbox layer."""
    bounds = MapBounds.from_config()
    bbox_gdf = gpd.GeoDataFrame(geometry=[bounds.geometry()], crs=MAP_CRS)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax)

    return fig, ax, bbox_gdf


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
    draw_continuous_colorbar(
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=LegendStyle(
            mode="continuous_inverted" if invert else "continuous",
            title=label,
            bottom_label=bottom_label,
            top_label=top_label,
            invert=invert,
            shrink=COLORBAR_KWARGS["shrink"],
            pad=COLORBAR_KWARGS["pad"],
            fraction=COLORBAR_KWARGS["fraction"],
        ),
    )


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


def prepare_bathymetry_values(
    gdf: gpd.GeoDataFrame,
    log_scale: bool,
) -> tuple[gpd.GeoDataFrame, str]:
    """Return bathymetry data and the column to plot."""
    if not log_scale:
        return gdf, "depth_m"

    out = gdf.copy()
    out["depth_log"] = np.log1p(out["depth_m"].clip(lower=0))
    return out, "depth_log"


def draw_h3_grid_outline(ax, grid: gpd.GeoDataFrame) -> None:
    """Draw a light H3 grid outline."""
    grid.plot(
        ax=ax,
        edgecolor=GRID_COLOR,
        facecolor="none",
        linewidth=0.1,
    )


def draw_bathymetry_legend(
    ax,
    cmap: str,
    norm: colors.Normalize,
    min_depth: float,
    max_depth: float,
) -> None:
    """Draw the inverted bathymetry legend."""
    draw_continuous_colorbar(
        ax,
        cmap,
        norm,
        legend=LegendStyle(
            mode="continuous_inverted",
            title="Depth",
            bottom_label=format_depth_label(max_depth),
            top_label=format_depth_label(min_depth),
        ),
    )


def draw_bathymetry_base_layer(
    ax,
    cmap: str = "Blues",
    legend: bool = True,
    vmax_quantile: float = 1,
    draw_grid: bool = False,
    log_scale: bool = False,
) -> gpd.GeoDataFrame:
    """Draw H3 bathymetry as a base layer."""
    static = load_static_features()
    grid = load_grid(uint64=True)
    plot_gdf, column = prepare_bathymetry_values(
        grid.merge(static, on="h3", how="left"),
        log_scale=log_scale,
    )

    values = plot_gdf[column].dropna()
    norm = bathymetry_norm(values, vmax_quantile)
    depth_values = plot_gdf["depth_m"].dropna()
    depth_norm = bathymetry_norm(depth_values, vmax_quantile)

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
        draw_bathymetry_legend(
            ax,
            cmap,
            norm,
            min_depth=depth_norm.vmin,
            max_depth=depth_norm.vmax,
        )

    if draw_grid:
        draw_h3_grid_outline(ax, grid)

    return plot_gdf


def draw_reference_layers(
    ax,
    bbox_gdf: gpd.GeoDataFrame,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw reference layers on top."""
    coast.plot(ax=ax, color=COAST_COLOR, linewidth=0.5)
    land.plot(ax=ax, color=LAND_COLOR, edgecolor="none")
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
    stroke = path_effects.withStroke(linewidth=1.2, foreground="#666666")
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
            "path_effects": [stroke],
        },
        zorder=10,
        path_effects=[stroke],
    )


def draw_reference_inset(
    ax,
    land: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
) -> None:
    """Draw a small global reference map with the region bbox."""
    inset_ax = ax.inset_axes([0.73, 0.025, 0.22, 0.18])
    inset_ax.set_facecolor(INSET_FACE_COLOR)

    land.plot(
        ax=inset_ax,
        color=INSET_LAND_COLOR,
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
