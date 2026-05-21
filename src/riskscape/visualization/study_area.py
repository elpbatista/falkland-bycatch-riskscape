"""Study area reference-layer map."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from riskscape.config import PROJECT_ROOT, cfg, paths
from riskscape.visualization.base_map import (
    INSET_FACE_COLOR,
    INSET_LAND_COLOR,
    draw_bathymetry_base_layer,
    draw_map_context,
    format_map_axes,
    load_reference_layers,
    setup_map,
)


STUDY_AREA_HIGHLIGHT_COLOR = "#c7352f"
STUDY_AREA_HIGHLIGHT_LINEWIDTH = 0.9


def load_study_area_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load fisheries grid and FICZ/FOCZ limits."""
    fisheries = gpd.read_file(PROJECT_ROOT / cfg["references"]["fisheries"])
    limits = gpd.read_file(PROJECT_ROOT / cfg["references"]["limits"])

    return fisheries.to_crs("EPSG:4326"), limits.to_crs("EPSG:4326")


def draw_highlighted_study_area(
    ax,
    bbox_gdf: gpd.GeoDataFrame,
) -> None:
    """Draw the study-area bbox as the main visual emphasis."""
    bbox_gdf.boundary.plot(
        ax=ax,
        edgecolor=STUDY_AREA_HIGHLIGHT_COLOR,
        linewidth=STUDY_AREA_HIGHLIGHT_LINEWIDTH,
        zorder=20,
    )


def draw_highlighted_reference_inset(
    ax,
    land: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
) -> None:
    """Draw an inset map with a highlighted study-area bbox."""
    inset_ax = ax.inset_axes([0.73, 0.025, 0.22, 0.18])
    inset_ax.set_facecolor(INSET_FACE_COLOR)

    land.plot(
        ax=inset_ax,
        color=INSET_LAND_COLOR,
        edgecolor="none",
        linewidth=0,
    )
    draw_highlighted_study_area(inset_ax, bbox_gdf)

    inset_ax.set_xlim(-180, 180)
    inset_ax.set_ylim(-90, 90)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#9a9a9a")
        spine.set_linewidth(0.6)


def plot_study_area_map(
    out_file: Path | None = None,
) -> Path:
    """Plot the configured study area reference layers."""
    title = "Falkland Islands Study Area"

    land, coast = load_reference_layers()
    fisheries, limits = load_study_area_layers()
    fig, ax, bbox_gdf = setup_map()

    draw_bathymetry_base_layer(
        ax,
        legend=True,
        draw_grid=True,
    )

    fisheries.plot(
        ax=ax,
        edgecolor="#f5a623",
        color="none",
        linewidth=0.45,
        alpha=0.75,
    )
    limits.plot(
        ax=ax,
        edgecolor="#f5a623",
        color="none",
        linewidth=1.1,
        alpha=0.95,
    )

    draw_map_context(
        ax,
        bbox_gdf,
        land,
        coast,
        show_reference_map=False,
    )
    draw_highlighted_study_area(ax, bbox_gdf)
    draw_highlighted_reference_inset(ax, land, bbox_gdf)
    format_map_axes(ax, title)

    if out_file is None:
        out_file = paths["plots"] / "study_area_reference_layers.png"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file
