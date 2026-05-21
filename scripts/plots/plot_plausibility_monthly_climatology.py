"""Plot monthly climatology matrices for environmental plausibility."""

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
from typing import cast

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
from riskscape.visualization.maps import plausibility_path


MODEL_NAME = "bayesian_gmm_k30"
PRODUCT_NAME = "joint"
YEARS = list(range(2014, 2024))
SPECIES = ["BBAL", "SAFS"]
OUTPUT_ROOT = paths["plots"] / "plausibility"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "plausibility"
VALUE_COL = "plausibility_non_zero_mean"
CMAP = "viridis"


def parse_years(value: str) -> list[int]:
    """Parse all, a range, or comma-separated years."""
    if value.lower() == "all":
        return YEARS

    years: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            years.update(range(int(start_text), int(end_text) + 1))
        else:
            years.add(int(item))

    if not years:
        raise ValueError("No years selected")

    return sorted(years)


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"

    return "_".join(str(year) for year in years)


def sql_list(files: list[Path]) -> str:
    """Return a DuckDB list literal for Parquet paths."""
    quoted = [f"'{str(path).replace("'", "''")}'" for path in files]

    return "[" + ", ".join(quoted) + "]"


def plausibility_files(
    years: list[int],
    model_name: str,
    product_name: str,
) -> list[Path]:
    """Return plausibility partition files for selected years."""
    files: list[Path] = []

    for year in years:
        path = plausibility_path(
            year=year,
            model_name=model_name,
            product_name=product_name,
        )

        if not path.exists():
            raise FileNotFoundError(f"Plausibility partition not found: {path}")

        files.append(path)

    return files


def monthly_climatology(
    years: list[int],
    model_name: str,
    product_name: str,
) -> pd.DataFrame:
    """Compute non-zero mean plausibility by month, species, and H3 cell."""
    files = plausibility_files(
        years=years,
        model_name=model_name,
        product_name=product_name,
    )
    query = f"""
        SELECT
            species,
            CAST(h3 AS UBIGINT) AS h3,
            CAST(month(date) AS INTEGER) AS month,
            CAST(avg(plausibility) AS FLOAT) AS {VALUE_COL},
            CAST(count(*) AS BIGINT) AS plausibility_non_zero_count
        FROM read_parquet({sql_list(files)})
        WHERE plausibility > 0
        GROUP BY species, h3, month
        ORDER BY species, month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(query).df()


def month_panel_title(month: int) -> str:
    """Return a compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    month: int,
    norm: colors.Normalize,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly plausibility panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask, ["h3", VALUE_COL]]
    plot_gdf = grid.merge(month_values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=[VALUE_COL])

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            column=VALUE_COL,
            cmap=CMAP,
            norm=norm,
            legend=False,
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


def plot_species_matrix(
    monthly: pd.DataFrame,
    species: str,
    years: list[int],
    output_root: Path,
) -> Path:
    """Plot a 3-by-4 monthly plausibility climatology matrix."""
    species_monthly = monthly[monthly["species"] == species].copy()

    if species_monthly.empty:
        raise ValueError(f"No monthly plausibility rows found for {species}")

    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

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

    label = year_label(years)
    fig.suptitle(
        f"Monthly Non-Zero Mean Environmental Plausibility — {species} — {label}",
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

    cax = fig.add_axes((0.88, 0.20, 0.025, 0.60))
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(CMAP)),
        cax=cax,
    )
    cbar.set_label("Non-zero mean plausibility")
    cbar.set_ticks([0.0, 1.0])
    cbar.ax.tick_params(which="major", length=2, labelsize=8)
    cbar.minorticks_off()
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    out_file = output_root / f"monthly_non_zero_mean_plausibility_{species}_{label}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot monthly plausibility climatology matrices.",
    )
    parser.add_argument(
        "--years",
        default="all",
        help="Use 'all', one year, a range like 2014-2023, or a comma list.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Plausibility model name.",
    )
    parser.add_argument(
        "--product",
        default=PRODUCT_NAME,
        help="Plausibility product name.",
    )
    parser.add_argument(
        "--species",
        default=",".join(SPECIES),
        help="Comma-separated species codes to plot.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated plausibility figures.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated plausibility summary exports.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the monthly plausibility climatology workflow."""
    args = parse_args()
    years = parse_years(args.years)
    species_values = [
        species.strip()
        for species in args.species.split(",")
        if species.strip()
    ]
    label = year_label(years)

    monthly = monthly_climatology(
        years=years,
        model_name=args.model,
        product_name=args.product,
    )

    summary_file = (
        args.data_output_root
        / f"monthly_non_zero_mean_plausibility_{label}.parquet"
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(summary_file, index=False)
    print("Saved:", summary_file)

    for species in species_values:
        out_file = plot_species_matrix(
            monthly=monthly,
            species=species,
            years=years,
            output_root=args.output_root,
        )
        print("Saved:", out_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
