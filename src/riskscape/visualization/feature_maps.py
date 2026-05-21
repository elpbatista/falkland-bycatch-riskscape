"""Feature map plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map


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
    plot_h3_map(
        gdf=plot_gdf,
        value_col=column,
        title=title,
        out_file=out_file,
        style=MapStyle(
            legend_mode="continuous",
            cmap=cmap,
            color_quantile=vmax_quantile,
            alpha_scale=False,
            hide_zero_values=False,
            show_reference_map=False,
            bathymetry=False,
        ),
    )
