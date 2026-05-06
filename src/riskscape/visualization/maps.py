"""Map model prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
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


# 0.5 vessel-hours per H3 cell/day
MINIMUM_EFFORT_UNIT = 0.5
MINIMUM_SPECIES_USE_UNIT = 0.0
MINIMUM_RISK_UNIT = np.log1p(MINIMUM_EFFORT_UNIT) + np.log1p(
    MINIMUM_SPECIES_USE_UNIT
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
    min_display_value: float | None = None
    color_quantile: float | None = 0.99
    alpha: float = 0.99
    alpha_min: float = 0.0
    alpha_gamma: float = 0.75
    alpha_scale: bool = True
    colorbar_labels: tuple[str, ...] | None = None
    colorbar_boundaries: tuple[float, ...] | None = None
    colorbar_quantiles: tuple[float, ...] | None = None


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

    if agg == "mean":
        out = out[out[value_col] > 0].copy()

    if out.empty:
        raise ValueError(f"No positive prediction rows found for {value_col}")

    grouped = (
        out.groupby("h3", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: f"{value_col}_{agg}"})
    )

    return grouped


def filter_prediction_rows(
    df: pd.DataFrame,
    species: str | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Return prediction rows for optional species and month filters."""
    out = df.copy()

    if species is not None:
        out = out[out["species"] == species]

    if month is not None:
        out = out[out["date"].dt.month == month]

    if out.empty:
        raise ValueError("No prediction rows found")

    return out


def summarize_hazard_h3(
    df: pd.DataFrame,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
    minimum_effort_unit: float = MINIMUM_EFFORT_UNIT,
) -> pd.DataFrame:
    """Summarize hazard as species use plus a fixed minimum effort unit by H3."""
    out = filter_prediction_rows(df, species=species, month=month)

    if agg == "mean":
        out = out[out["species_use_log_pred"] > 0].copy()

    if out.empty:
        raise ValueError("No positive species-use prediction rows found")

    grouped = out.groupby("h3", as_index=False).agg(
        species_use_log_pred=("species_use_log_pred", agg),
    )
    grouped["hazard_log_pred"] = (
        grouped["species_use_log_pred"] + np.log1p(minimum_effort_unit)
    )

    return grouped[["h3", "hazard_log_pred"]]


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

    if style.colorbar_labels is not None:
        return binned_color_norm(values, vmin, vmax, style)

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


def binned_color_norm(
    values: pd.Series,
    vmin: float,
    vmax: float,
    style: MapStyle,
) -> colors.BoundaryNorm:
    """Return a discrete color normalization for labeled map bands."""
    labels = style.colorbar_labels
    if labels is None:
        raise ValueError("Binned color norm requires colorbar labels")

    if style.colorbar_boundaries is not None and style.colorbar_quantiles is not None:
        raise ValueError("Use either colorbar boundaries or quantiles, not both")

    if style.colorbar_boundaries is not None:
        bins = np.asarray(style.colorbar_boundaries, dtype="float64")
        if len(bins) != len(labels) + 1:
            raise ValueError(
                "Colorbar boundaries must have one more value than labels"
            )
        if np.any(np.diff(bins) <= 0):
            raise ValueError("Colorbar boundaries must be strictly increasing")
    elif style.colorbar_quantiles is not None:
        quantiles = np.asarray(style.colorbar_quantiles, dtype="float64")
        if len(quantiles) != len(labels) + 1:
            raise ValueError(
                "Colorbar quantiles must have one more value than labels"
            )
        if np.any(np.diff(quantiles) <= 0):
            raise ValueError("Colorbar quantiles must be strictly increasing")
        if quantiles[0] < 0.0 or quantiles[-1] > 1.0:
            raise ValueError("Colorbar quantiles must be between 0 and 1")

        bins = values.quantile(quantiles).to_numpy(dtype="float64")
        if style.color_scale == "log":
            positive = values[values > 0]
            if positive.empty:
                raise ValueError("Log color scale requires positive values")
            bins[0] = max(bins[0], float(positive.min()))

        if np.any(np.diff(bins) <= 0):
            raise ValueError("Colorbar quantiles produced duplicate boundaries")
    elif style.color_scale == "log":
        positive = values[values > 0]
        if positive.empty:
            raise ValueError("Log color scale requires positive values")
        vmin = float(positive.min())
        if vmax <= vmin:
            vmax = vmin * 1.01
        bins = np.geomspace(vmin, vmax, len(labels) + 1)
    elif style.color_scale == "linear":
        if vmax <= vmin:
            vmax = vmin + 1.0
        bins = np.linspace(vmin, vmax, len(labels) + 1)
    else:
        raise ValueError(f"Unknown color scale: {style.color_scale}")

    return colors.BoundaryNorm(
        boundaries=bins,
        ncolors=plt.get_cmap(style.cmap).N,
        clip=True,
    )


def norm_limits(norm: colors.Normalize) -> tuple[float, float]:
    """Return lower and upper bounds for a color normalization."""
    if isinstance(norm, colors.BoundaryNorm):
        return float(norm.boundaries[0]), float(norm.boundaries[-1])

    return float(norm.vmin), float(norm.vmax)


def plottable_values(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    style: MapStyle,
) -> gpd.GeoDataFrame:
    """Return rows that should be drawn on the map."""
    plot_gdf = gdf.dropna(subset=[value_col]).copy()

    if style.min_display_value is not None:
        plot_gdf = plot_gdf[plot_gdf[value_col] > style.min_display_value].copy()
    elif style.hide_zero_values:
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
    vmin, vmax = norm_limits(norm)
    facecolors = cmap(norm(values.clip(lower=vmin, upper=vmax)))

    if style.alpha_scale:
        scaled = norm(values)
        if isinstance(norm, colors.BoundaryNorm):
            scaled = scaled / (plt.get_cmap(style.cmap).N - 1)
        scaled_values = pd.Series(scaled, index=values.index)
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
    if style.colorbar_labels is not None:
        draw_labeled_colorbar(ax, value_col, norm, style)
        return

    draw_compact_colorbar(
        ax=ax,
        cmap=style.cmap,
        norm=norm,
        label=legend_label(value_col),
    )


def draw_labeled_colorbar(
    ax,
    value_col: str,
    norm: colors.Normalize,
    style: MapStyle,
) -> None:
    """Draw a discrete colorbar with category labels."""
    if not isinstance(norm, colors.BoundaryNorm):
        raise TypeError("Labeled colorbar requires BoundaryNorm")

    boundaries = norm.boundaries
    ticks = (boundaries[:-1] + boundaries[1:]) / 2

    if style.color_scale == "log":
        ticks = np.sqrt(boundaries[:-1] * boundaries[1:])

    cbar = ax.figure.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(style.cmap)),
        ax=ax,
        label=legend_label(value_col),
        ticks=ticks,
        spacing="uniform",
        shrink=0.28,
        aspect=len(style.colorbar_labels),
        pad=0.02,
        fraction=0.035,
        drawedges=True,
    )
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("#8a8a8a")
    cbar.outline.set_linewidth(0.8)
    cbar.dividers.set_color("#8a8a8a")
    cbar.dividers.set_linewidth(0.6)
    cbar.ax.set_yticklabels(style.colorbar_labels)
    cbar.ax.tick_params(which="both", length=0)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("face")


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


def map_title(
    title: str,
    year: int | None = None,
    species: str | None = None,
    month: int | None = None,
    model_name: str | None = None,
    product_name: str | None = None,
) -> str:
    """Return a map title with prediction metadata appended when present."""
    parts = [title]

    model_parts = [
        part for part in (model_name, product_name) if part is not None
    ]
    if model_parts:
        parts.append("/".join(model_parts))

    if species is not None:
        parts.append(species)

    if year is not None:
        parts.append(str(year))

    if month is not None:
        parts.append(f"month {month:02d}")

    return " - ".join(parts)


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
    title: str | None = None,
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

    plot_title = map_title(
        title or f"{legend_label(value_col)} ({agg})",
        year=year,
        species=species,
        month=month,
        model_name=model_name,
        product_name=product_name,
    )

    out_file = (
        figure_root(model_name=model_name, product_name=product_name)
        / (
            f"{model_label}_{product_label}_{value_col}_{agg}_"
            f"{species_label}_{year}_{month_label}.png"
        )
    )

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=plot_title,
        out_file=out_file,
        style=style,
    )


def plot_hazard_map(
    year: int,
    species: str | None = None,
    month: int | None = None,
    agg: str = "mean",
    minimum_effort_unit: float = MINIMUM_EFFORT_UNIT,
    model_name: str | None = None,
    product_name: str | None = None,
    title: str | None = None,
    style: MapStyle | None = None,
) -> Path:
    """Plot hazard as species use plus a fixed minimum effort unit."""
    df = load_predictions(
        year=year,
        model_name=model_name,
        product_name=product_name,
    )

    summary = summarize_hazard_h3(
        df=df,
        species=species,
        month=month,
        agg=agg,
        minimum_effort_unit=minimum_effort_unit,
    )

    grid = load_grid(uint64=True)
    gdf = grid.merge(summary, on="h3", how="left")

    species_label, month_label, model_label, product_label = prediction_labels(
        species=species,
        month=month,
        model_name=model_name,
        product_name=product_name,
    )

    plot_title = map_title(
        title
        or (
            f"Hazard ({agg} species use + "
            f"{minimum_effort_unit:g} effort unit)"
        ),
        year=year,
        species=species,
        month=month,
        model_name=model_name,
        product_name=product_name,
    )

    out_file = (
        figure_root(model_name=model_name, product_name=product_name)
        / (
            f"{model_label}_{product_label}_hazard_log_pred_{agg}_"
            f"{species_label}_{year}_{month_label}.png"
        )
    )

    return plot_h3_map(
        gdf=gdf,
        value_col="hazard_log_pred",
        title=plot_title,
        out_file=out_file,
        style=style,
    )
