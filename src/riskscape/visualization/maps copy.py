"""Map model prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    draw_bathymetry_base_layer,
    draw_reference_layers,
    load_reference_layers,
    setup_map,
)


@dataclass(frozen=True)
class MapStyle:
    """Visual settings for prediction maps."""

    cmap: str = "YlOrRd"
    color_scale: str = "linear"
    bathymetry: bool = True
    bathymetry_log_scale: bool = False
    show_coordinates: bool = True
    show_north_arrow: bool = True
    show_reference_map: bool = True
    hide_zero_values: bool = True
    color_quantile: float | None = 0.99
    alpha: float = 0.95
    alpha_min: float = 0.0
    alpha_gamma: float = 0.75
    alpha_scale: bool = True


def prediction_path(
    year: int,
    model_name: str | None = None,
    product_name: str | None = None,
) -> Path:
    """Return prediction partition path."""
    path = paths["data"] / "modeling" / "predictions"

    if model_name is not None:
        path = path / model_name

    if product_name is not None:
        path = path / product_name

    return path / f"year={year}" / "part.parquet"


def figure_root(
    model_name: str | None = None,
    product_name: str | None = None,
) -> Path:
    """Return figure output directory."""
    path = paths["plots"] / "predictions"

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_predictions(
    year: int,
    model_name: str | None = None,
    product_name: str | None = None,
) -> pd.DataFrame:
    """Load prediction output for one year."""
    path = prediction_path(
        year=year,
        model_name=model_name,
        product_name=product_name,
    )

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    return pd.read_parquet(path)


def summarize_h3(
    df: pd.DataFrame,
    value_col: str,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Summarize prediction values by H3 cell."""
    out = df.copy()

    if species is not None:
        out = out[out["species"] == species]

    if month is not None:
        out = out[out["date"].dt.month == month]

    if out.empty:
        raise ValueError("No prediction rows found")

    grouped = (
        out.groupby("h3", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: f"{value_col}_{agg}"})
    )

    return grouped


def add_risk_class(
    gdf: gpd.GeoDataFrame,
    value_col: str,
) -> gpd.GeoDataFrame:
    """Add percentile risk class for nonzero values."""
    out = gdf.copy()
    values = out[value_col].dropna()

    if values.empty:
        raise ValueError(f"No values found for {value_col}")

    q90 = values.quantile(0.90)
    q95 = values.quantile(0.95)
    q99 = values.quantile(0.99)

    out["risk_class"] = "none"
    out.loc[out[value_col] > 0, "risk_class"] = "low"
    out.loc[out[value_col] >= q90, "risk_class"] = "high"
    out.loc[out[value_col] >= q95, "risk_class"] = "very_high"
    out.loc[out[value_col] >= q99, "risk_class"] = "extreme"

    return out


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


def legend_label(value_col: str) -> str:
    """Return a readable legend label for a plotted value column."""
    label = value_col.removesuffix("_mean")
    label = label.replace("_log_pred", " log prediction")
    label = label.replace("_pred", " prediction")
    label = label.replace("_", " ")
    return label.title()


def label_colorbar_extremes(fig, low: str = "Low", high: str = "High") -> None:
    """Label a GeoPandas colorbar with simple low/high endpoints."""
    if len(fig.axes) < 2:
        return

    cax = fig.axes[-1]
    cax.text(
        0.5,
        -0.03,
        low,
        transform=cax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    cax.text(
        0.5,
        1.03,
        high,
        transform=cax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
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
    bbox_gdf.boundary.plot(
        ax=inset_ax,
        edgecolor="#d62728",
        linewidth=1.2,
    )

    inset_ax.set_xlim(-180, 180)
    inset_ax.set_ylim(-90, 90)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#9a9a9a")
        spine.set_linewidth(0.6)


def scaled_alpha(
    values: pd.Series,
    vmin: float,
    vmax: float,
    style: MapStyle,
) -> np.ndarray:
    """Scale polygon opacity from value magnitude."""
    if vmax <= vmin:
        return np.full(len(values), style.alpha, dtype="float64")

    scaled = ((values - vmin) / (vmax - vmin)).clip(0, 1)
    scaled = scaled.pow(style.alpha_gamma)
    return (style.alpha_min + scaled * (style.alpha - style.alpha_min)).to_numpy()


def color_norm(
    values: pd.Series,
    style: MapStyle,
) -> colors.Normalize:
    """Return color normalization for plotted values."""
    vmin = 0.0 if values.min() >= 0 else values.min()
    vmax = values.max()

    if style.color_quantile is not None:
        vmax = values.quantile(style.color_quantile)
        if vmax <= vmin:
            vmax = values.max()

    if style.color_scale == "log":
        positive = values[values > 0]
        if positive.empty:
            raise ValueError("Log color scale requires positive values")
        return colors.LogNorm(vmin=positive.min(), vmax=vmax)

    if style.color_scale != "linear":
        raise ValueError(f"Unknown color scale: {style.color_scale}")

    return colors.Normalize(vmin=vmin, vmax=vmax)


def plot_h3_map(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    out_file: Path,
    style: MapStyle | None = None,
) -> Path:
    """Plot H3 map with bathymetry, land, and coastline."""
    style = style or MapStyle()
    land, coast = load_reference_layers()
    fig, ax, bbox_gdf = setup_map()

    if style.bathymetry:
        draw_bathymetry_base_layer(
            ax,
            legend=False,
            draw_grid=False,
            log_scale=style.bathymetry_log_scale,
        )

    plot_gdf = gdf.dropna(subset=[value_col]).copy()

    if style.hide_zero_values:
        plot_gdf = plot_gdf[plot_gdf[value_col] > 0].copy()

    if plot_gdf.empty:
        raise ValueError(f"No plottable values found for {value_col}")

    plot_values = plot_gdf[value_col].dropna()
    norm = color_norm(plot_values, style)
    cmap = plt.get_cmap(style.cmap)
    facecolors = cmap(norm(plot_gdf[value_col].clip(upper=norm.vmax)))
    if style.alpha_scale:
        facecolors[:, -1] = scaled_alpha(
            pd.Series(norm(plot_gdf[value_col]), index=plot_gdf.index),
            0.0,
            1.0,
            style,
        )
    else:
        facecolors[:, -1] = style.alpha

    plot_gdf.plot(
        ax=ax,
        color=facecolors,
        edgecolor="none",
        linewidth=0,
        missing_kwds={
            "color": "white",
            "alpha": 0.0,
        },
    )
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=legend_label(value_col),
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
    label_colorbar_extremes(fig)

    draw_reference_layers(ax, bbox_gdf, land, coast)

    if style.show_north_arrow:
        draw_north_arrow(ax)

    if style.show_reference_map:
        draw_reference_inset(ax, land, bbox_gdf)

    ax.set_title(title)
    if style.show_coordinates:
        format_coordinate_axes(ax)
    else:
        ax.set_axis_off()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def plot_prediction_map(
    year: int,
    value_col: str,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
    model_name: str | None = None,
    product_name: str | None = None,
    style: MapStyle | None = None,
) -> Path:
    """Plot summarized prediction map."""
    df = load_predictions(
        year=year,
        model_name=model_name,
        product_name=product_name,
    )

    summary = summarize_h3(
        df=df,
        value_col=value_col,
        species=species,
        month=month,
        agg=agg,
    )

    grid = load_grid(uint64=True)
    value_name = f"{value_col}_{agg}"

    gdf = grid.merge(summary, on="h3", how="left")

    species_label = species if species is not None else "all_species"
    month_label = f"month_{month:02d}" if month is not None else "all_months"
    model_label = model_name if model_name is not None else "default_model"
    product_label = product_name if product_name is not None else "default_product"

    title = (
        f"{value_col} ({agg}) - {species_label} - "
        f"{model_label}/{product_label} - {year} - {month_label}"
    )

    out_file = (
        figure_root(model_name=model_name, product_name=product_name)
        / f"{value_col}_{agg}_{species_label}_{year}_{month_label}.png"
    )

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=title,
        out_file=out_file,
        style=style,
    )
