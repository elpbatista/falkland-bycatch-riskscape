"""Plot weekly latent-risk climatology aggregated to fisheries grid squares."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import PROJECT_ROOT, cfg, paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    MAP_CRS,
    OCEAN_COLOR,
    MapBounds,
    draw_bathymetry_base_layer,
    draw_map_context,
    load_reference_layers,
)
from riskscape.visualization.maps import (
    MapStyle,
    draw_prediction_colorbar,
    draw_prediction_layer,
    plottable_values,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_weekly_operator_latent_risk import (  # noqa: E402
    END_YEAR,
    MODEL_NAME,
    PRODUCT_NAME,
    REPRESENTATIVE_WEEKS,
    SMALL_MULTIPLES_COLORBAR_POSITION,
    SPECIES,
    START_YEAR,
    climatology_path,
    load_weekly_climatology,
    risk_norm,
    risk_style,
)


OUTPUT_ROOT = paths["plots"] / "predictions" / "weekly_operator"
AGGREGATE_ROOT = paths["data"] / "plot_exports" / "weekly_operator"
GRID_LINE_COLOR = "#5f5f5f"
PROTECTION_ZONE_COLOR = "#404040"
VALUE_COL = "display_latent_risk_log_pred_mean"


def load_reference_overlays() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load fisheries grid squares and FICZ/FOCZ limits."""
    fisheries = gpd.read_file(PROJECT_ROOT / cfg["references"]["fisheries"])
    limits = gpd.read_file(PROJECT_ROOT / cfg["references"]["limits"])
    return fisheries.to_crs(MAP_CRS), limits.to_crs(MAP_CRS)


def fisheries_aggregate_path(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Return the fisheries-grid aggregate export path."""
    return (
        AGGREGATE_ROOT
        / model_name
        / product_name
        / (
            f"latent_risk_iso_week_climatology_{start_year}-{end_year}_"
            "fisheries_grid.parquet"
        )
    )


def output_path(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Return the fisheries-grid example figure path."""
    return (
        OUTPUT_ROOT
        / (
            f"{model_name}_{product_name}_latent_risk_iso_week_"
            f"climatology_{start_year}-{end_year}_fisheries_grid_example.png"
        )
    )


def build_h3_to_fisheries_lookup(
    h3_grid: gpd.GeoDataFrame,
    fisheries_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Assign H3 cells to fisheries squares by H3 centroid containment."""
    h3_points = h3_grid[["h3", "geometry"]].copy()
    h3_points["geometry"] = h3_points.geometry.representative_point()

    assigned = gpd.sjoin(
        h3_points,
        fisheries_grid[["group", "geometry"]],
        how="inner",
        predicate="within",
    )
    return assigned[["h3", "group"]].rename(columns={"group": "fisheries_grid"})


def aggregate_to_fisheries_grid(
    climatology: pd.DataFrame,
    h3_to_fisheries: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate H3 weekly climatology to fisheries grid squares."""
    joined = climatology.merge(h3_to_fisheries, on="h3", how="inner")
    return (
        joined.groupby(["fisheries_grid", "species", "iso_week"], as_index=False)
        .agg(
            n_h3=("h3", "nunique"),
            latent_risk_log_pred_mean=("latent_risk_log_pred_mean", "mean"),
            display_latent_risk_log_pred_mean=(
                "display_latent_risk_log_pred_mean",
                "mean",
            ),
        )
        .sort_values(["species", "iso_week", "fisheries_grid"])
    )


def plot_panel(
    ax: Axes,
    fisheries_grid: gpd.GeoDataFrame,
    limits: gpd.GeoDataFrame,
    aggregated: pd.DataFrame,
    species: str,
    week: int,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one fisheries-grid weekly climatology panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)
    draw_bathymetry_base_layer(ax, legend=False, draw_grid=False)

    values = aggregated[
        (aggregated["species"] == species) & (aggregated["iso_week"] == week)
    ][["fisheries_grid", VALUE_COL]]
    plot_gdf = fisheries_grid.merge(
        values,
        left_on="group",
        right_on="fisheries_grid",
        how="left",
    )
    plot_gdf = plot_gdf.dropna(subset=[VALUE_COL])

    if not plot_gdf.empty:
        try:
            plot_gdf = plottable_values(plot_gdf, VALUE_COL, style)
        except ValueError:
            plot_gdf = plot_gdf.iloc[0:0]

    if not plot_gdf.empty:
        draw_prediction_layer(
            ax=ax,
            gdf=plot_gdf,
            value_col=VALUE_COL,
            norm=norm,
            style=style,
        )

    fisheries_grid.boundary.plot(
        ax=ax,
        color=GRID_LINE_COLOR,
        linewidth=0.35,
        alpha=0.75,
        zorder=5,
    )
    limits.boundary.plot(
        ax=ax,
        color=PROTECTION_ZONE_COLOR,
        linewidth=1.0,
        alpha=0.95,
        zorder=6,
    )

    bbox_gdf = gpd.GeoDataFrame(geometry=[bounds.geometry()], crs=MAP_CRS)
    draw_map_context(
        ax,
        bbox_gdf,
        land,
        coast,
        show_north_arrow=False,
        show_reference_map=False,
    )
    ax.set_title(f"{species} - ISO week {week:02d}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_small_multiples(
    aggregated: pd.DataFrame,
    fisheries_grid: gpd.GeoDataFrame,
    limits: gpd.GeoDataFrame,
    species_values: list[str],
    weeks: list[int],
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
) -> Path:
    """Plot fisheries-grid weekly climatology example small multiples."""
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    style = risk_style(model_name, product_name, species_values)
    norm = risk_norm(style)

    fig, axes = plt.subplots(
        nrows=len(species_values),
        ncols=len(weeks),
        figsize=(4.0 * len(weeks), 5.1 * len(species_values)),
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)

    for row, species in enumerate(species_values):
        for col, week in enumerate(weeks):
            plot_panel(
                ax=cast(Axes, axes_array[row, col]),
                fisheries_grid=fisheries_grid,
                limits=limits,
                aggregated=aggregated,
                species=species,
                week=week,
                norm=norm,
                style=style,
                bounds=bounds,
                land=land,
                coast=coast,
            )

    fig.suptitle(
        f"Weekly Latent-Risk Climatology on Fisheries Grid - {start_year}-{end_year}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.91,
        top=0.93,
        bottom=0.035,
        wspace=0.09,
        hspace=0.16,
    )

    cax = fig.add_axes(SMALL_MULTIPLES_COLORBAR_POSITION)
    draw_prediction_colorbar(
        ax=cast(Axes, axes_array[0, -1]),
        value_col=VALUE_COL,
        norm=norm,
        style=style,
        cax=cast(Axes, cax),
    )

    out_file = output_path(
        model_name=model_name,
        product_name=product_name,
        start_year=start_year,
        end_year=end_year,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weekly latent-risk climatology on fisheries grid.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--species", nargs="+", default=list(SPECIES))
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=list(REPRESENTATIVE_WEEKS),
    )
    return parser.parse_args()


def main() -> int:
    """Run fisheries-grid weekly climatology example plot."""
    args = parse_args()
    climatology = load_weekly_climatology(
        climatology_path(
            model_name=args.model_name,
            product_name=args.product_name,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    )
    h3_grid = load_grid(uint64=True)
    fisheries_grid, limits = load_reference_overlays()
    h3_to_fisheries = build_h3_to_fisheries_lookup(h3_grid, fisheries_grid)
    aggregated = aggregate_to_fisheries_grid(climatology, h3_to_fisheries)

    aggregate_path = fisheries_aggregate_path(
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(aggregate_path, index=False)
    print(f"Saved aggregate: {aggregate_path}")

    out_file = plot_small_multiples(
        aggregated=aggregated,
        fisheries_grid=fisheries_grid,
        limits=limits,
        species_values=list(args.species),
        weeks=list(args.weeks),
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
