"""Plot dominant Bayesian/GMM environmental component assignments."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
import duckdb
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
from typing import Any, cast

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
MODEL_NAME = "bayesian_gmm_k30"
PRODUCT_NAME = "joint"
INPUT_ROOT = (
    paths["data"]
    / "modeling"
    / "environmental_regimes"
)
OUTPUT_ROOT = paths["plots"] / "plausibility"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "plausibility"

COMPONENT_COLORS = {
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
    """Return a stable discrete color lookup for component classes."""
    color_lookup = dict(COMPONENT_COLORS)
    palettes = ["tab20", "tab20b", "tab20c"]
    colors_needed = max(0, n_classes - len(color_lookup))

    if colors_needed == 0:
        return color_lookup

    generated: list[str] = []
    for palette_name in palettes:
        cmap = colormaps[palette_name]
        generated.extend(
            colors.to_hex(cmap(i / cmap.N))
            for i in range(cmap.N)
        )

    for component in range(len(color_lookup), n_classes):
        color_lookup[component] = generated[
            (component - len(COMPONENT_COLORS)) % len(generated)
        ]

    return color_lookup


def component_colors(components: list[int]) -> dict[int, str]:
    """Return colors for the observed components."""
    if not components:
        return {}
    return extended_class_colors(max(components) + 1)


def draw_component_colorbar(fig: Any, cax: Axes, components: list[int]) -> None:
    """Draw a discrete component color bar matching the risk-map style."""
    if not components:
        cax.axis("off")
        return

    lookup = component_colors(components)
    cmap = colors.ListedColormap([lookup[component] for component in components])
    boundaries = list(range(len(components) + 1))
    ticks = [idx + 0.5 for idx in range(len(components))]
    norm = colors.BoundaryNorm(boundaries, cmap.N)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        label="Component",
        ticks=ticks,
        spacing="uniform",
        drawedges=True,
    )
    cbar.ax.set_yticklabels([str(component) for component in components])
    cbar.ax.tick_params(which="both", length=0, labelsize=8)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cast(Any, cbar.solids).set_edgecolor("face")


def component_path(year: int, input_root: Path) -> Path:
    """Return component-assignment partition path for a year."""
    return input_root / f"year={year}" / "part.parquet"


def dominant_components(
    year: int,
    model_name: str,
    product_name: str,
    input_root: Path,
) -> pd.DataFrame:
    """Return dominant component per H3 cell from environmental features."""
    _ = (model_name, product_name)
    component_file = component_path(year, input_root)

    if not component_file.exists():
        raise FileNotFoundError(
            f"Component assignment partition not found: {component_file}"
        )

    query = """
        WITH environmental_rows AS (
            SELECT DISTINCT
                CAST(h3 AS UBIGINT) AS h3,
                date,
                bayesian_gmm_k30_component AS component,
                bayesian_gmm_k30_component_probability AS component_probability,
                bayesian_gmm_k30_component_entropy AS component_entropy
            FROM read_parquet(?)
        ),
        counts AS (
            SELECT
                h3,
                component,
                count(*) AS component_days,
                avg(component_probability) AS mean_component_probability,
                avg(component_entropy) AS mean_component_entropy
            FROM environmental_rows
            GROUP BY h3, component
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3
                    ORDER BY component_days DESC, mean_component_probability DESC, component
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            CAST(component AS INTEGER) AS dominant_component,
            component_days,
            mean_component_probability,
            mean_component_entropy
        FROM ranked
        WHERE rank = 1
        ORDER BY h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            [str(component_file)],
        ).df()


def monthly_dominant_components(
    year: int,
    model_name: str,
    product_name: str,
    input_root: Path,
) -> pd.DataFrame:
    """Return dominant component by month/H3 from environmental features."""
    _ = (model_name, product_name)
    component_file = component_path(year, input_root)

    if not component_file.exists():
        raise FileNotFoundError(
            f"Component assignment partition not found: {component_file}"
        )

    query = """
        WITH environmental_rows AS (
            SELECT DISTINCT
                CAST(h3 AS UBIGINT) AS h3,
                date,
                bayesian_gmm_k30_component AS component,
                bayesian_gmm_k30_component_probability AS component_probability,
                bayesian_gmm_k30_component_entropy AS component_entropy
            FROM read_parquet(?)
        ),
        counts AS (
            SELECT
                h3,
                CAST(month(date) AS INTEGER) AS month,
                component,
                count(*) AS component_days,
                avg(component_probability) AS mean_component_probability,
                avg(component_entropy) AS mean_component_entropy
            FROM environmental_rows
            GROUP BY h3, month, component
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY h3, month
                    ORDER BY component_days DESC, mean_component_probability DESC, component
                ) AS rank
            FROM counts
        )
        SELECT
            h3,
            month,
            CAST(component AS INTEGER) AS dominant_component,
            component_days,
            mean_component_probability,
            mean_component_entropy
        FROM ranked
        WHERE rank = 1
        ORDER BY month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            [str(component_file)],
        ).df()


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_component_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> list[int]:
    """Draw one dominant-component map panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    if summary.empty:
        raise ValueError("No component rows found")

    plot_gdf = grid.merge(summary, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=["dominant_component"]).copy()
    plot_gdf["dominant_component"] = plot_gdf["dominant_component"].astype(int)
    lookup = component_colors(plot_gdf["dominant_component"].unique().tolist())
    plot_gdf["component_color"] = plot_gdf["dominant_component"].map(
        lookup
    )

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf["component_color"],
            edgecolor="none",
            linewidth=0,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bounds.geometry()],
        crs=grid.crs or MAP_CRS,
    )
    draw_reference_layers(ax, bbox_gdf, land, coast)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    return sorted(plot_gdf["dominant_component"].unique().tolist())


def save_component_map(
    summary: pd.DataFrame,
    year: int,
    out_file: Path,
) -> None:
    """Save one dominant-component map."""
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=False)
    used_components = plot_component_panel(
        ax=ax,
        grid=grid,
        summary=summary,
        bounds=bounds,
        land=land,
        coast=coast,
    )

    fig.suptitle(
        f"Dominant Bayesian/GMM Environmental Components — {year}",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.03,
        right=0.86,
        top=0.92,
        bottom=0.04,
    )
    components = sorted(used_components)
    cbar_width = 0.030
    fig_width, fig_height = fig.get_size_inches()
    segment_height = cbar_width * fig_width / fig_height
    cbar_height = segment_height * max(1, len(components))
    cbar_bottom = 0.50 - cbar_height / 2
    cbar_rect: tuple[float, float, float, float] = (
        0.89,
        cbar_bottom,
        cbar_width,
        cbar_height,
    )
    cax = fig.add_axes(cbar_rect)
    draw_component_colorbar(fig, cax, components)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    month: int,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> list[int]:
    """Draw one monthly dominant-component panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask].copy()

    if month_values.empty:
        plot_gdf = grid.iloc[0:0].copy()
    else:
        plot_gdf = grid.merge(month_values, on="h3", how="inner")
        plot_gdf["dominant_component"] = plot_gdf["dominant_component"].astype(int)
        lookup = component_colors(plot_gdf["dominant_component"].unique().tolist())
        plot_gdf["component_color"] = plot_gdf["dominant_component"].map(
            lookup
        )

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf["component_color"],
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

    return sorted(plot_gdf["dominant_component"].unique().tolist())


def save_monthly_component_matrix(
    monthly: pd.DataFrame,
    year: int,
    out_file: Path,
) -> None:
    """Save a 12-panel monthly dominant-component matrix."""
    if monthly.empty:
        raise ValueError("No monthly component rows found")

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
    used_components: set[int] = set()

    for month, ax in enumerate(axes_flat, start=1):
        used_components.update(
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
        f"Monthly Dominant Bayesian/GMM Environmental Components — {year}",
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
    components = sorted(used_components)
    cbar_width = 0.025
    fig_width, fig_height = fig.get_size_inches()
    segment_height = cbar_width * fig_width / fig_height
    cbar_height = segment_height * max(1, len(components))
    cbar_bottom = 0.50 - cbar_height / 2
    cbar_rect: tuple[float, float, float, float] = (
        0.88,
        cbar_bottom,
        cbar_width,
        cbar_height,
    )
    cax = fig.add_axes(cbar_rect)
    draw_component_colorbar(fig, cax, components)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dominant Bayesian/GMM environmental components.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model", default=MODEL_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--product", default=PRODUCT_NAME, help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=INPUT_ROOT,
        help="Directory containing component assignment year=*/part.parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated component map figures.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated component summary exports.",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Also generate 12-panel monthly component matrices.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the component mapping workflow."""
    args = parse_args()

    summary = dominant_components(
        year=args.year,
        model_name=args.model,
        product_name=args.product,
        input_root=args.input_root,
    )
    summary_file = (
        args.data_output_root
        / f"dominant_bayesian_gmm_components_{args.year}.parquet"
    )

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_file, index=False)

    print("Saved:", summary_file)

    figure_file = args.output_root / f"dominant_bayesian_gmm_components_{args.year}.png"
    save_component_map(
        summary=summary,
        year=args.year,
        out_file=figure_file,
    )
    print("Saved:", figure_file)

    if args.monthly:
        monthly = monthly_dominant_components(
            year=args.year,
            model_name=args.model,
            product_name=args.product,
            input_root=args.input_root,
        )
        monthly_file = (
            args.data_output_root
            / f"monthly_dominant_bayesian_gmm_components_{args.year}.parquet"
        )
        monthly.to_parquet(monthly_file, index=False)
        print("Saved:", monthly_file)

        figure_file = (
            args.output_root
            / f"monthly_dominant_bayesian_gmm_components_{args.year}.png"
        )
        save_monthly_component_matrix(
            monthly=monthly,
            year=args.year,
            out_file=figure_file,
        )
        print("Saved:", figure_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
