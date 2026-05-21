"""Shared helpers for categorical H3 component and seascape maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, cast

import geopandas as gpd
from matplotlib import colors, colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    MAP_CRS,
    MapBounds,
    OCEAN_COLOR,
    draw_reference_layers,
    load_reference_layers,
)
from riskscape.visualization.maps import MapStyle, plot_h3_map
from riskscape.visualization.monthly_maps import (
    MonthlyMapLayout,
    add_centered_colorbar_axis,
    create_monthly_map_grid,
    format_month_panel,
    month_axes,
    save_monthly_map,
)


DEFAULT_CLASS_COLORS = {
    0: "#4e79a7",
    1: "#f28e2b",
    2: "#e15759",
    3: "#76b7b2",
    4: "#59a14f",
    5: "#edc948",
    6: "#b07aa1",
    7: "#ff9da7",
    8: "#9c755f",
    9: "#bab0ac",
}


def extended_class_colors(
    n_classes: int,
    base_colors: dict[int, str] | None = None,
) -> dict[int, str]:
    """Return a stable discrete color lookup for integer classes."""
    color_lookup = dict(base_colors or DEFAULT_CLASS_COLORS)
    colors_needed = max(0, n_classes - len(color_lookup))

    if colors_needed == 0:
        return color_lookup

    generated: list[str] = []
    for palette_name in ("tab20", "tab20b", "tab20c"):
        cmap = colormaps[palette_name]
        generated.extend(colors.to_hex(cmap(i / cmap.N)) for i in range(cmap.N))

    base_count = len(color_lookup)
    for class_value in range(base_count, n_classes):
        color_lookup[class_value] = generated[(class_value - base_count) % len(generated)]

    return color_lookup


def observed_class_colors(
    class_values: Iterable[int],
    base_colors: dict[int, str] | None = None,
) -> dict[int, str]:
    """Return colors for observed integer classes."""
    values = list(class_values)
    if not values:
        return {}
    return extended_class_colors(max(values) + 1, base_colors=base_colors)


def draw_categorical_colorbar(
    fig: plt.Figure,
    cax: Axes,
    class_values: list[int],
    label: str,
    base_colors: dict[int, str] | None = None,
) -> None:
    """Draw a compact discrete colorbar for integer classes."""
    if not class_values:
        cax.axis("off")
        return

    lookup = observed_class_colors(class_values, base_colors=base_colors)
    cmap = colors.ListedColormap([lookup[value] for value in class_values])
    boundaries = list(range(len(class_values) + 1))
    ticks = [idx + 0.5 for idx in range(len(class_values))]
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label=label,
        ticks=ticks,
        spacing="uniform",
        drawedges=True,
    )
    cbar.ax.set_yticklabels([str(value) for value in class_values])
    cbar.ax.tick_params(which="both", length=0, labelsize=8)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cast(Any, cbar.solids).set_edgecolor("face")


def categorical_map_style(
    class_values: list[int],
    colorbar_label: str,
    base_colors: dict[int, str] | None = None,
) -> MapStyle:
    """Return a shared MapStyle for integer categorical maps."""
    if not class_values:
        raise ValueError("Categorical map style requires at least one class")

    classes = sorted(class_values)
    lookup = observed_class_colors(classes, base_colors=base_colors)
    boundaries = tuple(
        float(value)
        for value in (
            [classes[0] - 0.5]
            + [(left + right) / 2 for left, right in zip(classes, classes[1:])]
            + [classes[-1] + 0.5]
        )
    )
    return MapStyle(
        legend_mode="categorical",
        colorbar_title=colorbar_label,
        color_palette=tuple(lookup[value] for value in classes),
        colorbar_labels=tuple(str(value) for value in classes),
        colorbar_boundaries=boundaries,
        bathymetry=False,
        show_reference_map=False,
        hide_zero_values=False,
        alpha_scale=False,
        alpha=1.0,
    )


def draw_categorical_h3_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    values: pd.DataFrame,
    value_col: str,
    color_col: str,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
    title: str | None = None,
    base_colors: dict[int, str] | None = None,
) -> list[int]:
    """Draw one categorical H3 panel and return the classes used."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if values.empty:
        plot_gdf = grid.iloc[0:0].copy()
    else:
        plot_gdf = grid.merge(values, on="h3", how="inner")
        plot_gdf[value_col] = plot_gdf[value_col].astype(int)
        lookup = observed_class_colors(plot_gdf[value_col].unique().tolist(), base_colors)
        plot_gdf[color_col] = plot_gdf[value_col].map(lookup)

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf[color_col],
            edgecolor="none",
            linewidth=0,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bounds.geometry()],
        crs=grid.crs or MAP_CRS,
    )
    draw_reference_layers(ax, bbox_gdf, land, coast)

    if title is not None:
        ax.set_title(title, fontsize=11)

    if plot_gdf.empty:
        return []

    return sorted(plot_gdf[value_col].unique().tolist())


def save_single_categorical_map(
    values: pd.DataFrame,
    value_col: str,
    colorbar_label: str,
    title: str,
    out_file: Path,
    base_colors: dict[int, str] | None = None,
) -> Path:
    """Save one categorical H3 map."""
    if values.empty:
        raise ValueError("No categorical rows found")

    grid = load_grid(uint64=True)
    gdf = grid.merge(values, on="h3", how="inner")
    gdf[value_col] = gdf[value_col].astype(int)
    classes = sorted(gdf[value_col].dropna().unique().tolist())
    return plot_h3_map(
        gdf=gdf,
        value_col=value_col,
        title=title,
        out_file=out_file,
        style=categorical_map_style(
            class_values=classes,
            colorbar_label=colorbar_label,
            base_colors=base_colors,
        ),
    )


def save_monthly_categorical_matrix(
    monthly: pd.DataFrame,
    value_col: str,
    colorbar_label: str,
    title: str,
    out_file: Path,
    base_colors: dict[int, str] | None = None,
    layout: MonthlyMapLayout | None = None,
) -> Path:
    """Save a 12-panel monthly categorical H3 matrix."""
    if monthly.empty:
        raise ValueError("No monthly categorical rows found")

    layout = layout or MonthlyMapLayout()
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    fig, axes = create_monthly_map_grid(title, layout=layout)
    used_classes: set[int] = set()

    for month, ax in month_axes(axes):
        format_month_panel(ax, month=month, bounds=bounds, layout=layout)
        month_mask = cast(pd.Series, monthly["month"]).eq(month)
        used_classes.update(
            draw_categorical_h3_panel(
                ax=ax,
                grid=grid,
                values=monthly.loc[month_mask].copy(),
                value_col=value_col,
                color_col=f"{value_col}_color",
                bounds=bounds,
                land=land,
                coast=coast,
                title=None,
                base_colors=base_colors,
            )
        )

    classes = sorted(used_classes)
    cax = add_centered_colorbar_axis(fig, max(1, len(classes)), layout=layout)
    draw_categorical_colorbar(
        fig,
        cax,
        classes,
        label=colorbar_label,
        base_colors=base_colors,
    )
    return save_monthly_map(fig, out_file, layout=layout)
