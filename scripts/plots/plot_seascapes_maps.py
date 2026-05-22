"""Plot dominant seascape assignments."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
from matplotlib import colormaps
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
MODEL_NAME = "som_15x15_hierarchical_k30"
DEFAULT_ASSIGNMENT_TABLE = "environmental_regimes"
OUTPUT_ROOT = paths["plots"] / "seascapes"
CLASS_COLUMN = "seascape"
SCORE_COLUMN = "seascape_distance"
SCORE_ORDER = "asc"
FILE_PREFIX = "monthly_dominant_som_hierarchical_seascapes"
TITLE_LABEL = "SOM-Hierarchical Seascapes"

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


def extended_class_colors(n_classes: int) -> dict[int, str]:
    """Return a stable discrete color lookup for seascape classes."""
    color_lookup = dict(SEASCAPE_COLORS)
    palettes = ["tab20", "tab20b", "tab20c"]
    colors_needed = max(0, n_classes - len(color_lookup))

    if colors_needed == 0:
        return color_lookup

    generated: list[str] = []
    for palette_name in palettes:
        cmap = colormaps[palette_name]
        generated.extend(colors.to_hex(cmap(i / cmap.N)) for i in range(cmap.N))

    for seascape in range(len(color_lookup), n_classes):
        color_lookup[seascape] = generated[
            (seascape - len(SEASCAPE_COLORS)) % len(generated)
        ]

    return color_lookup


def seascape_colors(seascapes: list[int]) -> dict[int, str]:
    """Return colors for observed seascapes."""
    if not seascapes:
        return {}
    return extended_class_colors(max(seascapes) + 1)


def draw_seascape_colorbar(fig: Any, cax: Axes, seascapes: list[int]) -> None:
    """Draw a discrete seascape color bar matching the original map style."""
    if not seascapes:
        cax.axis("off")
        return

    lookup = seascape_colors(seascapes)
    cmap = colors.ListedColormap([lookup[seascape] for seascape in seascapes])
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


def assignment_path(year: int, model_name: str, assignment_table: str | None) -> Path:
    """Return seascape assignment partition path for one year."""
    table = assignment_table or DEFAULT_ASSIGNMENT_TABLE
    return (
        paths["data"]
        / "modeling"
        / table
        / f"year={year}"
        / "part.parquet"
    )


def quote_identifier(name: str) -> str:
    """Return a DuckDB-safe identifier."""
    return '"' + name.replace('"', '""') + '"'


def monthly_dominant_seascapes(
    year: int,
    model_name: str,
    assignment_table: str | None = None,
    class_column: str = CLASS_COLUMN,
    score_column: str | None = SCORE_COLUMN,
    score_order: str = SCORE_ORDER,
    drop_class: list[int] | None = None,
) -> pd.DataFrame:
    """Return dominant seascape by month/H3."""
    seascape_file = assignment_path(year, model_name, assignment_table)

    if not seascape_file.exists():
        raise FileNotFoundError(
            f"Seascape assignment partition not found: {seascape_file}"
        )

    class_expr = quote_identifier(class_column)
    score_select = ""
    score_order_expr = ""
    where_clause = ""

    if drop_class:
        dropped = ", ".join(str(int(value)) for value in drop_class)
        where_clause = f"WHERE {class_expr} NOT IN ({dropped})"

    if score_column:
        score_expr = quote_identifier(score_column)
        score_select = f", avg({score_expr}) AS mean_seascape_score"
        direction = "ASC" if score_order == "asc" else "DESC"
        score_order_expr = f", mean_seascape_score {direction}"

    query = """
        WITH counts AS (
            SELECT
                CAST(h3 AS UBIGINT) AS h3,
                CAST(month(date) AS INTEGER) AS month,
                {class_expr} AS seascape,
                count(*) AS seascape_days
                {score_select}
            FROM read_parquet(?)
            {where_clause}
            GROUP BY h3, month, {class_expr}
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3, month
                    ORDER BY seascape_days DESC{score_order_expr}, seascape
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            month,
            CAST(seascape AS INTEGER) AS dominant_seascape,
            seascape_days
            {score_output}
        FROM ranked
        WHERE rank = 1
        ORDER BY month, h3
    """.format(
        class_expr=class_expr,
        where_clause=where_clause,
        score_select=score_select,
        score_order_expr=score_order_expr,
        score_output=", mean_seascape_score" if score_column else "",
    )

    with duckdb.connect(database=":memory:") as con:
        return con.execute(query, [str(seascape_file)]).df()


def display_model_name(model_name: str) -> str:
    """Return display-safe model text for figure titles."""
    if "_k" in model_name:
        return f"k = {model_name.rsplit('_k', maxsplit=1)[1]}"
    return model_name


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
        lookup = seascape_colors(plot_gdf["dominant_seascape"].unique().tolist())
        plot_gdf["seascape_color"] = plot_gdf["dominant_seascape"].map(lookup)

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
    title_label: str = TITLE_LABEL,
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
        f"Monthly Dominant {title_label} ({display_model_name(model_name)}) — {year}",
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
    colorbar_bottom = 0.50 - cbar_height / 2
    cax = fig.add_axes(
        (
            0.88,
            colorbar_bottom,
            cbar_width,
            cbar_height,
        )
    )
    draw_seascape_colorbar(fig, cax, seascapes)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dominant seascape assignments.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--assignment-table",
        help=(
            "Modeling table containing yearly seascape assignments. "
            f"Defaults to {DEFAULT_ASSIGNMENT_TABLE}."
        ),
    )
    parser.add_argument("--class-column", default=CLASS_COLUMN)
    parser.add_argument("--score-column", default=SCORE_COLUMN)
    parser.add_argument(
        "--drop-class",
        type=int,
        action="append",
        default=[],
        help="Class value to exclude before summarizing. May be repeated.",
    )
    parser.add_argument(
        "--score-order",
        default=SCORE_ORDER,
        choices=("asc", "desc"),
        help="Tie-break order for the monthly mean score column.",
    )
    parser.add_argument("--file-prefix", default=FILE_PREFIX)
    parser.add_argument("--title-label", default=TITLE_LABEL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated seascape map figures.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the seascape mapping workflow."""
    args = parse_args()
    monthly = monthly_dominant_seascapes(
        year=args.year,
        model_name=args.model_name,
        assignment_table=args.assignment_table,
        class_column=args.class_column,
        score_column=args.score_column,
        score_order=args.score_order,
        drop_class=args.drop_class,
    )
    figure_file = (
        args.output_root
        / f"{args.file_prefix}_{args.model_name}_{args.year}.png"
    )
    save_monthly_seascape_matrix(
        monthly=monthly,
        year=args.year,
        model_name=args.model_name,
        out_file=figure_file,
        title_label=args.title_label,
    )
    print("Saved:", figure_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
