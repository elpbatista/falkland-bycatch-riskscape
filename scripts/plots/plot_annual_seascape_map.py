"""Plot annual dominant seascape assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from plot_seascapes_maps import (
    CLASS_COLUMN,
    DEFAULT_ASSIGNMENT_TABLE,
    MODEL_NAME,
    OUTPUT_ROOT,
    SCORE_COLUMN,
    SCORE_ORDER,
    TITLE_LABEL,
    assignment_path,
    display_model_name,
    draw_seascape_colorbar,
    quote_identifier,
    seascape_colors,
)
from riskscape.grid import load_grid
from riskscape.visualization.base_map import (
    MAP_CRS,
    draw_map_context,
    format_map_axes,
    load_reference_layers,
    setup_map,
)


YEAR = 2022
FILE_PREFIX = "annual_dominant_som_hierarchical_seascapes"


def annual_dominant_seascapes(
    year: int,
    model_name: str,
    assignment_table: str | None = None,
    class_column: str = CLASS_COLUMN,
    score_column: str | None = SCORE_COLUMN,
    score_order: str = SCORE_ORDER,
    drop_class: list[int] | None = None,
) -> pd.DataFrame:
    """Return dominant annual seascape by H3 cell."""
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
                {class_expr} AS seascape,
                count(*) AS seascape_days
                {score_select}
            FROM read_parquet(?)
            {where_clause}
            GROUP BY h3, {class_expr}
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3
                    ORDER BY seascape_days DESC{score_order_expr}, seascape
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            CAST(seascape AS INTEGER) AS dominant_seascape,
            seascape_days
            {score_output}
        FROM ranked
        WHERE rank = 1
        ORDER BY h3
    """.format(
        class_expr=class_expr,
        where_clause=where_clause,
        score_select=score_select,
        score_order_expr=score_order_expr,
        score_output=", mean_seascape_score" if score_column else "",
    )

    with duckdb.connect(database=":memory:") as con:
        return con.execute(query, [str(seascape_file)]).df()


def save_annual_seascape_map(
    annual: pd.DataFrame,
    year: int,
    model_name: str,
    out_file: Path,
    title_label: str = TITLE_LABEL,
    show_all_classes: bool = True,
) -> None:
    """Save a single annual dominant-seascape map with matrix colorbar styling."""
    if annual.empty:
        raise ValueError("No annual seascape rows found")

    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    fig, ax, bbox_gdf = setup_map()
    plot_gdf = grid.merge(annual, on="h3", how="inner")
    if plot_gdf.empty:
        raise ValueError("No seascape rows intersected the H3 grid")

    plot_gdf["dominant_seascape"] = plot_gdf["dominant_seascape"].astype(int)
    used_seascapes = sorted(plot_gdf["dominant_seascape"].unique().tolist())
    legend_classes = list(range(30)) if show_all_classes else used_seascapes
    lookup = seascape_colors(legend_classes)
    plot_gdf["seascape_color"] = plot_gdf["dominant_seascape"].map(lookup)

    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["seascape_color"],
        edgecolor="none",
        linewidth=0,
    )

    if bbox_gdf.crs is None:
        bbox_gdf = gpd.GeoDataFrame(geometry=bbox_gdf.geometry, crs=grid.crs or MAP_CRS)

    draw_map_context(
        ax,
        bbox_gdf,
        land,
        coast,
        show_north_arrow=True,
        show_reference_map=False,
    )
    format_map_axes(
        ax,
        (
            f"Annual Dominant {title_label} "
            f"({display_model_name(model_name)}) - {year}"
        ),
        show_coordinates=True,
    )
    fig.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.08)
    cbar_width = 0.025
    fig_width, fig_height = fig.get_size_inches()
    segment_height = cbar_width * fig_width / fig_height
    cbar_height = segment_height * max(1, len(legend_classes))
    ax_box = ax.get_position()
    cbar_bottom = ax_box.y0 + (ax_box.height - cbar_height) / 2
    cax = fig.add_axes((ax_box.x1 + 0.012, cbar_bottom, cbar_width, cbar_height))
    draw_seascape_colorbar(fig, cax, legend_classes)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot annual dominant seascape assignments.",
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
        help="Tie-break order for the annual mean score column.",
    )
    parser.add_argument("--file-prefix", default=FILE_PREFIX)
    parser.add_argument("--title-label", default=TITLE_LABEL)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated seascape map figures.",
    )
    parser.add_argument(
        "--observed-classes-only",
        action="store_true",
        help="Show only classes present as annual dominant classes in the colorbar.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the annual seascape mapping workflow."""
    args = parse_args()
    annual = annual_dominant_seascapes(
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
    save_annual_seascape_map(
        annual=annual,
        year=args.year,
        model_name=args.model_name,
        out_file=figure_file,
        title_label=args.title_label,
        show_all_classes=not args.observed_classes_only,
    )
    print("Saved:", figure_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
