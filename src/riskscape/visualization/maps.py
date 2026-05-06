"""Map model prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    draw_bathymetry_base_layer,
    draw_compact_colorbar,
    draw_map_context,
    format_map_axes,
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
    alpha: float = 0.99
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


def legend_label(value_col: str) -> str:
    """Return a readable legend label for a plotted value column."""
    label = value_col.removesuffix("_mean")
    label = label.replace("_log_pred", " log prediction")
    label = label.replace("_pred", " prediction")
    label = label.replace("_", " ")
    return label.title()


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


def plottable_values(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    style: MapStyle,
) -> gpd.GeoDataFrame:
    """Return rows that should be drawn on the map."""
    plot_gdf = gdf.dropna(subset=[value_col]).copy()

    if style.hide_zero_values:
        plot_gdf = plot_gdf[plot_gdf[value_col] > 0].copy()

    if plot_gdf.empty:
        raise ValueError(f"No plottable values found for {value_col}")

    return plot_gdf


def map_facecolors(
    values: pd.Series,
    norm: colors.Normalize,
    style: MapStyle,
) -> np.ndarray:
    """Return RGBA face colors for the H3 polygons."""
    cmap = plt.get_cmap(style.cmap)
    facecolors = cmap(norm(values.clip(upper=norm.vmax)))

    if style.alpha_scale:
        scaled_values = pd.Series(norm(values), index=values.index)
        facecolors[:, -1] = scaled_alpha(scaled_values, 0.0, 1.0, style)
    else:
        facecolors[:, -1] = style.alpha

    return facecolors


def draw_prediction_layer(
    ax,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    norm: colors.Normalize,
    style: MapStyle,
) -> None:
    """Draw the prediction polygons."""
    gdf.plot(
        ax=ax,
        color=map_facecolors(gdf[value_col], norm, style),
        edgecolor="none",
        linewidth=0,
    )


def draw_prediction_colorbar(
    ax,
    value_col: str,
    norm: colors.Normalize,
    style: MapStyle,
) -> None:
    """Draw a compact low/high colorbar."""
    draw_compact_colorbar(
        ax=ax,
        cmap=style.cmap,
        norm=norm,
        label=legend_label(value_col),
    )


def prediction_labels(
    species: str | None,
    month: int | None,
    model_name: str | None,
    product_name: str | None,
) -> tuple[str, str, str, str]:
    """Return display-safe labels for prediction metadata."""
    return (
        species if species is not None else "all_species",
        f"month_{month:02d}" if month is not None else "all_months",
        model_name if model_name is not None else "default_model",
        product_name if product_name is not None else "default_product",
    )


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

    plot_gdf = plottable_values(gdf, value_col, style)
    norm = color_norm(plot_gdf[value_col].dropna(), style)

    draw_prediction_layer(ax, plot_gdf, value_col, norm, style)
    draw_prediction_colorbar(ax, value_col, norm, style)
    draw_map_context(
        ax,
        bbox_gdf,
        land,
        coast,
        show_north_arrow=style.show_north_arrow,
        show_reference_map=style.show_reference_map,
    )
    format_map_axes(
        ax,
        title,
        show_coordinates=style.show_coordinates,
    )

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

    species_label, month_label, model_label, product_label = prediction_labels(
        species=species,
        month=month,
        model_name=model_name,
        product_name=product_name,
    )

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
