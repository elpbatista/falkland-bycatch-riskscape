"""Plot dominant feature-only KMeans seascape assignments."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
from typing import Any, cast

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
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


YEAR = 2022
MODEL_NAME = "kmeans_k10"
INPUT_ROOT = paths["data"] / "modeling" / "seascapes"
OUTPUT_ROOT = paths["plots"] / "seascapes"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "seascapes"

SEASCAPE_COLORS = {
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


def draw_seascape_colorbar(fig: Any, cax: Axes, seascapes: list[int]) -> None:
    """Draw a discrete seascape color bar matching the component-map style."""
    if not seascapes:
        cax.axis("off")
        return

    cmap = colors.ListedColormap([SEASCAPE_COLORS[seascape] for seascape in seascapes])
    boundaries = list(range(len(seascapes) + 1))
    ticks = [idx + 0.5 for idx in range(len(seascapes))]
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label="Seascape",
        ticks=ticks,
        spacing="uniform",
        drawedges=True,
    )
    cbar.ax.set_yticklabels([str(seascape) for seascape in seascapes])
    cbar.ax.tick_params(which="both", length=0, labelsize=8)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cast(Any, cbar.solids).set_edgecolor("face")


def seascape_path(year: int, model_name: str) -> Path:
    """Return seascape assignment partition path for one year."""
    return INPUT_ROOT / model_name / f"year={year}" / "part.parquet"


def monthly_dominant_seascapes(year: int, model_name: str) -> pd.DataFrame:
    """Return dominant seascape by month/H3."""
    seascape_file = seascape_path(year, model_name)

    if not seascape_file.exists():
        raise FileNotFoundError(
            f"Seascape assignment partition not found: {seascape_file}"
        )

    query = """
        WITH counts AS (
            SELECT
                CAST(h3 AS UBIGINT) AS h3,
                CAST(month(date) AS INTEGER) AS month,
                seascape,
                count(*) AS seascape_days,
                avg(seascape_distance) AS mean_seascape_distance
            FROM read_parquet(?)
            GROUP BY h3, month, seascape
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3, month
                    ORDER BY seascape_days DESC, mean_seascape_distance ASC, seascape
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            month,
            CAST(seascape AS INTEGER) AS dominant_seascape,
            seascape_days,
            mean_seascape_distance
        FROM ranked
        WHERE rank = 1
        ORDER BY month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(query, [str(seascape_file)]).df()


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    month: int,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> list[int]:
    """Draw one monthly dominant-seascape panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask].copy()

    if month_values.empty:
        plot_gdf = grid.iloc[0:0].copy()
    else:
        plot_gdf = grid.merge(month_values, on="h3", how="inner")
        plot_gdf["dominant_seascape"] = plot_gdf["dominant_seascape"].astype(int)
        plot_gdf["seascape_color"] = plot_gdf["dominant_seascape"].map(
            SEASCAPE_COLORS
        )

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf["seascape_color"],
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

    if plot_gdf.empty:
        return []

    return sorted(plot_gdf["dominant_seascape"].unique().tolist())


def save_monthly_seascape_matrix(
    monthly: pd.DataFrame,
    year: int,
    model_name: str,
    out_file: Path,
) -> None:
    """Save a 12-panel monthly dominant-seascape matrix."""
    if monthly.empty:
        raise ValueError("No monthly seascape rows found")

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
    used_seascapes: set[int] = set()

    for month, ax in enumerate(axes_flat, start=1):
        used_seascapes.update(
            plot_month_panel(
                ax=ax,
                grid=grid,
                monthly=monthly,
                month=month,
                bounds=bounds,
                land=land,
                coast=coast,
            )
        )

    fig.suptitle(
        f"Monthly Dominant KMeans Seascapes — {year}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.84,
        top=0.95,
        bottom=0.035,
        wspace=0.15,
        hspace=0.15,
    )
    seascapes = sorted(used_seascapes)
    cbar_width = 0.025
    fig_width, fig_height = fig.get_size_inches()
    segment_height = cbar_width * fig_width / fig_height
    cbar_height = segment_height * max(1, len(seascapes))
    cbar_bottom = 0.50 - cbar_height / 2
    cbar_rect: tuple[float, float, float, float] = (
        0.88,
        cbar_bottom,
        cbar_width,
        cbar_height,
    )
    cax = fig.add_axes(cbar_rect)
    draw_seascape_colorbar(fig, cax, seascapes)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dominant feature-only KMeans seascapes.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated seascape map figures.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated seascape summary exports.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the seascape mapping workflow."""
    args = parse_args()
    monthly = monthly_dominant_seascapes(
        year=args.year,
        model_name=args.model_name,
    )
    monthly_file = (
        args.data_output_root
        / f"monthly_dominant_kmeans_seascapes_{args.model_name}_{args.year}.parquet"
    )
    monthly_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(monthly_file, index=False)
    print("Saved:", monthly_file)

    figure_file = (
        args.output_root
        / f"monthly_dominant_kmeans_seascapes_{args.model_name}_{args.year}.png"
    )
    save_monthly_seascape_matrix(
        monthly=monthly,
        year=args.year,
        model_name=args.model_name,
        out_file=figure_file,
    )
    print("Saved:", figure_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
