"""Plot weekly latent risk with fishing-activity cells marked."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TypeAlias, cast

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiLineString, MultiPolygon, Polygon

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
    add_risk_colorbar,
    climatology_path,
    load_weekly_climatology,
    risk_norm,
    risk_style,
)
from plot_weekly_operator_fisheries_grid_example import (  # noqa: E402
    aggregate_to_fisheries_grid,
    build_h3_to_fisheries_lookup,
    load_reference_overlays,
    plot_panel,
)
from riskscape.config import paths  # noqa: E402
from riskscape.grid import load_grid  # noqa: E402
from riskscape.visualization.base_map import MapBounds, load_reference_layers  # noqa: E402


YEAR = 2022
FLAG: str | None = None
OUTPUT_ROOT = paths["plots"] / "fishing_activity"
OVERLAY_COLOR = "#202020"
LineSegment: TypeAlias = list[tuple[float, ...]]


def exposure_path(year: int) -> Path:
    """Return the gear/flag fishing activity export for one year."""
    return (
        paths["data"]
        / "plot_exports"
        / "fishing_activity"
        / f"fishing_effort_by_gear_flag_{year}.parquet"
    )


def load_weekly_flag_h3(year: int, flag: str | None) -> pd.DataFrame:
    """Return H3 fishing activity by ISO week before grid aggregation."""
    path = exposure_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Gear/flag exposure table not found: {path}")

    query = """
        SELECT
            CAST(h3 AS UBIGINT) AS h3,
            CAST(date_part('isoyear', CAST(date AS DATE)) AS INTEGER) AS iso_year,
            CAST(date_part('week', CAST(date AS DATE)) AS INTEGER) AS iso_week,
            SUM(fishing_hours)::FLOAT AS fishing_hours,
            SUM(vessel_count)::FLOAT AS vessel_count
        FROM read_parquet($path)
        WHERE ($flag IS NULL OR flag = $flag)
        GROUP BY h3, iso_year, iso_week
        HAVING iso_year = $year AND SUM(fishing_hours) > 0
        ORDER BY iso_week, h3
    """
    with duckdb.connect(database=":memory:") as con:
        out = con.execute(
            query,
            {"path": str(path), "flag": flag, "year": year},
        ).fetchdf()

    out["h3"] = out["h3"].astype("uint64")
    return out


def aggregate_flag_to_fisheries_grid(
    flag_h3: pd.DataFrame,
    h3_to_fisheries: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate flag-filtered fishing activity to fisheries cells by ISO week."""
    joined = flag_h3.merge(h3_to_fisheries, on="h3", how="inner")
    return (
        joined.groupby(["fisheries_grid", "iso_week"], as_index=False)
        .agg(
            n_h3=("h3", "nunique"),
            fishing_hours=("fishing_hours", "sum"),
            vessel_count=("vessel_count", "sum"),
        )
        .sort_values(["iso_week", "fisheries_grid"])
    )


def overlay_flag_fisheries_cells(
    ax: Axes,
    fisheries_grid: gpd.GeoDataFrame,
    flag_cells: pd.DataFrame,
    week: int,
) -> None:
    """Mark fisheries cells with flag-filtered activity for one ISO week."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    values = flag_cells[flag_cells["iso_week"] == week][
        ["fisheries_grid"]
    ].drop_duplicates()
    if values.empty:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return

    overlay = fisheries_grid.merge(
        values,
        left_on="group",
        right_on="fisheries_grid",
        how="inner",
    )
    if overlay.empty:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return

    lines = geometry_lines(overlay.geometry)
    if not lines:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return

    collection = LineCollection(
        lines,
        colors=OVERLAY_COLOR,
        linewidth=1.15,
        alpha=0.98,
        zorder=7,
        clip_on=True,
    )
    ax.add_collection(collection, autolim=False)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def geometry_lines(geometries: gpd.GeoSeries) -> list[LineSegment]:
    """Return exterior line segments without changing axes limits."""
    lines: list[LineSegment] = []
    for geometry in geometries:
        if isinstance(geometry, Polygon):
            lines.append(list(geometry.exterior.coords))
        elif isinstance(geometry, MultiPolygon):
            for polygon in geometry.geoms:
                lines.append(list(polygon.exterior.coords))
        elif isinstance(geometry, MultiLineString):
            for line in geometry.geoms:
                lines.append(list(line.coords))
    return lines


def output_path(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
    year: int,
    flag: str | None,
) -> Path:
    """Return output figure path."""
    overlay_label = flag.lower() if flag is not None else "all_vessel"
    return (
        OUTPUT_ROOT
        / (
            f"{model_name}_{product_name}_latent_risk_iso_week_"
            f"climatology_{start_year}-{end_year}_{overlay_label}_cells_{year}.png"
        )
    )


def plot_small_multiples(
    aggregated_latent_risk: pd.DataFrame,
    fisheries_grid: gpd.GeoDataFrame,
    limits: gpd.GeoDataFrame,
    flag_cells: pd.DataFrame,
    species_values: list[str],
    weeks: list[int],
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
    year: int,
    flag: str | None,
) -> Path:
    """Plot fisheries-grid weekly latent risk with activity cells marked."""
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
            ax = cast(Axes, axes_array[row, col])
            plot_panel(
                ax=ax,
                fisheries_grid=fisheries_grid,
                limits=limits,
                aggregated=aggregated_latent_risk,
                species=species,
                week=week,
                norm=norm,
                style=style,
                bounds=bounds,
                land=land,
                coast=coast,
            )
            overlay_flag_fisheries_cells(ax, fisheries_grid, flag_cells, week)

    overlay_label = flag if flag is not None else "All Vessel"
    fig.suptitle(
        f"Weekly Latent-Risk Climatology with {overlay_label} Fisheries Cells - {year}",
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
    add_risk_colorbar(
        ax=cast(Axes, axes_array[0, -1]),
        norm=norm,
        style=style,
        cax=cast(Axes, cax),
    )

    out_file = output_path(model_name, product_name, start_year, end_year, year, flag)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weekly latent risk with fishing-activity cells marked.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--species", nargs="+", default=list(SPECIES))
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=list(REPRESENTATIVE_WEEKS),
    )
    parser.add_argument(
        "--flag",
        default=FLAG,
        help="Optional flag filter, such as FLK. Omit to mark all vessel activity.",
    )
    return parser.parse_args()


def main() -> int:
    """Run weekly latent-risk maps with fishing activity marked."""
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
    aggregated_latent_risk = aggregate_to_fisheries_grid(
        climatology,
        h3_to_fisheries,
    )
    flag_h3 = load_weekly_flag_h3(args.year, args.flag)
    flag_cells = aggregate_flag_to_fisheries_grid(flag_h3, h3_to_fisheries)
    out_file = plot_small_multiples(
        aggregated_latent_risk=aggregated_latent_risk,
        fisheries_grid=fisheries_grid,
        limits=limits,
        flag_cells=flag_cells,
        species_values=list(args.species),
        weeks=list(args.weeks),
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
        year=args.year,
        flag=args.flag,
    )
    print("Saved:", out_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
