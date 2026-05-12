"""Plot 12-panel monthly latent-risk matrices from prediction outputs."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
from typing import cast

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
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
    SPECIES_USE_LOG_COLOR_MAX,
    SPECIES_USE_LOG_MIN_DISPLAY,
    aggregation_name,
)


YEAR = 2022
MODEL_NAME = "hybrid_presence_gate_extra_trees_kmeans_k15_blockcv_bayesian_gmm_k30"
PRODUCT_NAME = "joint"
SPECIES = ["BBAL", "SAFS"]
AGG = "non_zero_mean"
CMAP = "YlOrRd"
RISK_ALPHA = 0.90
OUTPUT_ROOT = paths["plots"] / "predictions"
LATENT_RISK_BASELINE = float(np.log1p(MINIMUM_EFFORT_UNIT))
LATENT_RISK_MIN_DISPLAY = LATENT_RISK_BASELINE


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


def shared_log_norm(values: pd.Series, vmax_quantile: float) -> colors.LogNorm:
    """Return a fixed framework color scale for all panels and species."""
    positive = values.dropna()
    if positive.empty:
        raise ValueError("Log color scale requires positive latent-risk values")

    vmin = LATENT_RISK_MIN_DISPLAY
    vmax = float(SPECIES_USE_LOG_COLOR_MAX + np.log1p(MINIMUM_EFFORT_UNIT))

    if vmax <= vmin:
        vmax = float(positive.max())
    if vmax <= vmin:
        vmax = vmin * 1.01

    return colors.LogNorm(vmin=vmin, vmax=vmax)


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    month: int,
    norm: colors.LogNorm,
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
        plot_gdf.plot(
            ax=ax,
            column="latent_risk_log_pred",
            cmap=CMAP,
            norm=norm,
            legend=False,
            alpha=RISK_ALPHA,
            edgecolor="none",
            linewidth=0,
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
    norm: colors.LogNorm,
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

    cax = fig.add_axes((0.88, 0.20, 0.025, 0.60))
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP)),
        cax=cax,
    )
    cbar.set_label("Latent Risk")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)

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
        "--vmax-quantile",
        type=float,
        default=0.99,
        help="Upper quantile for the shared log color scale.",
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
    norm = shared_log_norm(
        cast(pd.Series, monthly["latent_risk_log_pred"]),
        vmax_quantile=args.vmax_quantile,
    )

    for species in args.species:
        out_file = plot_monthly_matrix(
            monthly=monthly,
            species=species,
            year=args.year,
            model_name=args.model_name,
            product_name=args.product_name,
            agg=args.agg,
            norm=norm,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
