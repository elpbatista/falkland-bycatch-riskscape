"""Feature map plotting utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    draw_reference_layers,
    load_reference_layers,
    setup_map,
)


def plot_h3_feature_map(
    df: pd.DataFrame,
    column: str,
    out_file: Path,
    title: str,
    cmap: str = "viridis",
    vmax_quantile: float = 0.98,
) -> None:
    """Plot an H3 feature map."""
    grid = load_grid(uint64=True)
    plot_gdf = grid.merge(df, on="h3", how="left")

    land, coast = load_reference_layers()
    _, ax, bbox_gdf = setup_map()

    values = plot_gdf[column]
    vmax = values.quantile(vmax_quantile)

    plot_gdf.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=True,
        edgecolor="none",
        linewidth=0,
        vmax=vmax,
    )

    grid.plot(
        ax=ax,
        edgecolor="darkgrey",
        facecolor="none",
        linewidth=0.1,
    )

    draw_reference_layers(ax, bbox_gdf, land, coast)

    ax.set_title(title)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()