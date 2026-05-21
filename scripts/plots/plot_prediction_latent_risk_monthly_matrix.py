"""Plot 12-panel monthly latent-risk matrices from prediction outputs."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import calendar
from dataclasses import replace
from pathlib import Path
import sys
from typing import cast

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    MAP_CRS,
    MapBounds,
    OCEAN_COLOR,
    draw_reference_layers,
    load_reference_layers,
)
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_MIN_DISPLAY,
    aggregation_name,
    color_norm,
    draw_prediction_colorbar,
    draw_prediction_layer,
    plottable_values,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_prediction_maps import (  # noqa: E402
    aggregate_expression,
    shared_styles,
)


YEAR = 2022
MODEL_NAME = "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30"
PRODUCT_NAME = "joint"
SPECIES = ["BBAL", "SAFS"]
AGG = "non_zero_mean"
OUTPUT_ROOT = paths["plots"] / "predictions"
LATENT_RISK_BASELINE = float(np.log1p(MINIMUM_EFFORT_UNIT))
COLORBAR_WIDTH = 0.025
DEFAULT_COLOR_RANGE_FACTOR = 1.0
DEFAULT_MATRIX_COLOR_QUANTILES = (0.0, 0.40, 0.75, 0.90, 1.0)


def prediction_path(year: int, model_name: str, product_name: str) -> Path:
    """Return one prediction partition path."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / model_name
        / product_name
        / f"year={year}"
        / "part.parquet"
    )


def aggregate_expression(column: str, agg: str) -> str:
    """Return a DuckDB expression matching the map aggregation mode."""
    agg_name = aggregation_name(agg)

    if agg_name == "non_zero_median":
        return (
            f"COALESCE(median(CASE WHEN {column} > 0 THEN {column} "
            "ELSE NULL END), 0.0)"
        )
    if agg_name == "non_zero_mean":
        return (
            f"COALESCE(avg(CASE WHEN {column} > 0 THEN {column} "
            "ELSE NULL END), 0.0)"
        )

    raise ValueError("--agg must be non_zero_median or non_zero_mean")


def monthly_latent_risk(
    year: int,
    model_name: str,
    product_name: str,
    species_values: list[str],
    agg: str,
) -> pd.DataFrame:
    """Return monthly H3 latent-risk summaries for selected species."""
    path = prediction_path(year, model_name, product_name)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    species_sql = ", ".join(f"'{species}'" for species in species_values)
    use_expr = aggregate_expression("species_use_log_pred", agg)

    query = f"""
        SELECT
            h3,
            species,
            CAST(month(date) AS INTEGER) AS month,
            {use_expr}::FLOAT AS species_use_log_pred,
            ({use_expr} + {LATENT_RISK_BASELINE})::FLOAT AS latent_risk_log_pred
        FROM read_parquet($path, hive_partitioning=false)
        WHERE species IN ({species_sql})
        GROUP BY h3, species, month
        ORDER BY species, month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        out = con.execute(query, {"path": str(path)}).fetchdf()

    out["h3"] = out["h3"].astype("uint64")
    return out


def shared_latent_risk_style(
    year: int,
    model_name: str,
    product_name: str,
    species_values: list[str],
    agg: str,
):
    """Return the same binned latent-risk style used by prediction maps."""
    path = prediction_path(year, model_name, product_name)
    species_use_expr = aggregate_expression("species_use_log_pred", agg)
    risk_expr = aggregate_expression("risk_log_pred", agg)
    frames = []

    with duckdb.connect(database=":memory:") as con:
        for species in species_values:
            query = f"""
                SELECT
                    h3,
                    species,
                    {species_use_expr}::FLOAT AS species_use_log_pred,
                    {risk_expr}::FLOAT AS risk_log_pred
                FROM read_parquet($path, hive_partitioning=false)
                WHERE species = $species
                GROUP BY h3, species
            """
            frames.append(
                con.execute(
                    query,
                    {"path": str(path), "species": species},
                ).fetchdf()
            )

    predictions = pd.concat(frames, ignore_index=True)
    predictions["h3"] = predictions["h3"].astype("uint64")
    _, _, hazard_style, _ = shared_styles(
        predictions=predictions,
        species=species_values,
        agg=agg,
    )
    return hazard_style


def stretched_binned_style(style, factor: float):
    """Return the same binned style with a wider experimental color range."""
    if factor == DEFAULT_COLOR_RANGE_FACTOR:
        return style
    if factor <= 0:
        raise ValueError("--color-range-factor must be greater than 0")
    if style.colorbar_boundaries is None:
        return style

    bins = np.asarray(style.colorbar_boundaries, dtype="float64")
    if bins.size < 2:
        return style

    lower = float(bins[0])
    upper = float(bins[-1])
    stretched_upper = lower + (upper - lower) * factor
    if stretched_upper <= lower:
        return style

    if style.color_scale == "log" and lower > 0:
        stretched_bins = np.geomspace(lower, stretched_upper, bins.size)
    else:
        stretched_bins = np.linspace(lower, stretched_upper, bins.size)

    return replace(
        style,
        colorbar_boundaries=tuple(float(value) for value in stretched_bins),
        color_max=float(stretched_bins[-1]),
    )


def quantile_binned_style(
    style,
    values: pd.Series,
    quantiles: tuple[float, ...],
):
    """Return the same binned style with boundaries from supplied values."""
    if style.colorbar_labels is None:
        return style
    if len(quantiles) != len(style.colorbar_labels) + 1:
        raise ValueError(
            "--color-quantiles must provide one more value than colorbar labels"
        )
    if any(value < 0 or value > 1 for value in quantiles):
        raise ValueError("--color-quantiles values must be between 0 and 1")
    if any(next_value < value for value, next_value in zip(quantiles, quantiles[1:])):
        raise ValueError("--color-quantiles values must be sorted")

    lower = style.color_min if style.color_min is not None else 0.0
    positive = values[values > lower].dropna()
    if positive.empty:
        return style

    bins = positive.quantile(quantiles).to_numpy(dtype="float64")
    bins[0] = lower

    if (bins[1:] <= bins[:-1]).any():
        upper = float(positive.max())
        if upper <= lower:
            upper = lower * 1.01 if lower > 0 else 1.0
        if style.color_scale == "log" and lower > 0:
            bins = np.geomspace(lower, upper, len(style.colorbar_labels) + 1)
        else:
            bins = np.linspace(lower, upper, len(style.colorbar_labels) + 1)

    return replace(
        style,
        colorbar_quantiles=None,
        colorbar_boundaries=tuple(float(value) for value in bins),
        color_max=float(bins[-1]),
    )


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    month: int,
    norm: colors.Normalize,
    style,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly latent-risk panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[
        month_mask,
        ["h3", "species_use_log_pred", "latent_risk_log_pred"],
    ]
    plot_gdf = grid.merge(month_values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=["species_use_log_pred", "latent_risk_log_pred"])
    plot_gdf = plot_gdf[
        plot_gdf["species_use_log_pred"] > SPECIES_USE_LOG_MIN_DISPLAY
    ].copy()

    if not plot_gdf.empty:
        plot_gdf = plottable_values(
            plot_gdf,
            "latent_risk_log_pred",
            style,
        )
        draw_prediction_layer(
            ax,
            plot_gdf,
            value_col="latent_risk_log_pred",
            norm=norm,
            style=style,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bounds.geometry()],
        crs=grid.crs or MAP_CRS,
    )

    draw_reference_layers(ax, bbox_gdf, land, coast)
    ax.set_title(month_panel_title(month), fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def output_path(
    model_name: str,
    product_name: str,
    species: str,
    year: int,
    agg: str,
) -> Path:
    """Return output file path."""
    return (
        OUTPUT_ROOT
        / f"{model_name}_{product_name}_latent_risk_log_pred_{agg}_"
        f"{species}_{year}_monthly_matrix.png"
    )


def plot_monthly_matrix(
    monthly: pd.DataFrame,
    species: str,
    year: int,
    model_name: str,
    product_name: str,
    agg: str,
    norm: colors.Normalize,
    style,
) -> Path:
    """Plot a 3-by-4 monthly latent-risk matrix."""
    species_monthly = monthly[monthly["species"] == species].copy()
    if species_monthly.empty:
        raise ValueError(f"No monthly latent-risk rows found for {species}")

    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(10.5, 16),
        constrained_layout=False,
    )
    axes_flat = cast(list[Axes], axes.ravel().tolist())

    for month, ax in enumerate(axes_flat, start=1):
        plot_month_panel(
            ax=ax,
            grid=grid,
            monthly=species_monthly,
            month=month,
            norm=norm,
            style=style,
            bounds=bounds,
            land=land,
            coast=coast,
        )

    fig.suptitle(f"Monthly Latent Risk — {species}, {year}", fontsize=16, y=0.985)
    fig.subplots_adjust(
        left=0.025,
        right=0.84,
        top=0.95,
        bottom=0.035,
        wspace=0.15,
        hspace=0.15,
    )

    fig_width, fig_height = fig.get_size_inches()
    segment_height = COLORBAR_WIDTH * fig_width / fig_height
    colorbar_height = segment_height * len(style.colorbar_labels or ())
    colorbar_bottom = 0.50 - colorbar_height / 2
    cax = fig.add_axes(
        (
            0.88,
            colorbar_bottom,
            COLORBAR_WIDTH,
            colorbar_height,
        )
    )
    draw_prediction_colorbar(
        axes_flat[-1],
        value_col="latent_risk_log_pred",
        norm=norm,
        style=style,
        cax=cax,
    )

    out_file = output_path(
        model_name=model_name,
        product_name=product_name,
        species=species,
        year=year,
        agg=agg,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create 3-by-4 monthly latent-risk matrices.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--species", nargs="+", default=SPECIES)
    parser.add_argument(
        "--agg",
        default=AGG,
        choices=("non_zero_median", "non_zero_mean"),
    )
    parser.add_argument(
        "--color-range-factor",
        type=float,
        default=DEFAULT_COLOR_RANGE_FACTOR,
        help=(
            "Experiment-only multiplier for the shared binned color range. "
            "Use 1.0 to match prediction maps exactly."
        ),
    )
    parser.add_argument(
        "--color-bin-source",
        choices=("risk_map", "monthly_pooled", "monthly_species"),
        default="risk_map",
        help=(
            "Source for binned color boundaries. risk_map matches prediction "
            "maps; monthly_* are experiment-only tail-emphasis modes."
        ),
    )
    parser.add_argument(
        "--color-quantiles",
        nargs=5,
        type=float,
        default=DEFAULT_MATRIX_COLOR_QUANTILES,
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4"),
        help="Quantiles used by monthly_pooled/monthly_species bin sources.",
    )
    return parser.parse_args()


def main() -> int:
    """Run monthly latent-risk plotting."""
    args = parse_args()
    monthly = monthly_latent_risk(
        year=args.year,
        model_name=args.model_name,
        product_name=args.product_name,
        species_values=list(args.species),
        agg=args.agg,
    )
    style = shared_latent_risk_style(
        year=args.year,
        model_name=args.model_name,
        product_name=args.product_name,
        species_values=list(args.species),
        agg=args.agg,
    )
    style = stretched_binned_style(style, args.color_range_factor)
    color_quantiles = tuple(float(value) for value in args.color_quantiles)

    if args.color_bin_source == "monthly_pooled":
        style = quantile_binned_style(
            style,
            cast(pd.Series, monthly["latent_risk_log_pred"]),
            color_quantiles,
        )

    for species in args.species:
        species_style = style
        if args.color_bin_source == "monthly_species":
            species_mask = cast(pd.Series, monthly["species"]).eq(species)
            species_style = quantile_binned_style(
                style,
                cast(pd.Series, monthly.loc[species_mask, "latent_risk_log_pred"]),
                color_quantiles,
            )
        norm = color_norm(
            cast(pd.Series, monthly["latent_risk_log_pred"]),
            species_style,
        )
        out_file = plot_monthly_matrix(
            monthly=monthly,
            species=species,
            year=args.year,
            model_name=args.model_name,
            product_name=args.product_name,
            agg=args.agg,
            norm=norm,
            style=species_style,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
