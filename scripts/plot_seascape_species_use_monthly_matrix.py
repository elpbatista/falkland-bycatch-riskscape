"""Plot monthly seascape-conditioned species-use maps."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
from typing import cast

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
SPECIES = ["BBAL", "SAFS"]
VALUE_COL = "seascape_non_zero_median_residence_index"
AGG = "non_zero_median"
INPUT_ROOT = paths["data"] / "modeling" / "seascape_species_use"
OUTPUT_ROOT = paths["plots"] / "seascapes" / "species_use"
CMAP = "YlOrRd"


def surface_path(year: int, model_name: str) -> Path:
    """Return seascape-conditioned species-use surface path."""
    return INPUT_ROOT / model_name / f"year={year}" / "part.parquet"


def load_surface(year: int, model_name: str) -> pd.DataFrame:
    """Load seascape-conditioned species-use columns for one year."""
    path = surface_path(year, model_name)

    if not path.exists():
        raise FileNotFoundError(f"Seascape species-use surface not found: {path}")

    return pd.read_parquet(
        path,
        columns=["h3", "date", "species", VALUE_COL],
    )


def normalize_agg(agg: str) -> str:
    """Normalize accepted aggregation names."""
    if agg == "non_zero_media":
        return "non_zero_median"
    return agg


def summarize_monthly_species_use(
    df: pd.DataFrame,
    species: str,
    agg: str = AGG,
) -> pd.DataFrame:
    """Summarize seascape-conditioned species use by month and H3 cell."""
    agg = normalize_agg(agg)
    if agg not in {"non_zero_median", "non_zero_mean"}:
        raise ValueError("--agg must be non_zero_median or non_zero_mean")

    out = df[df["species"] == species].dropna(subset=["h3", "date", VALUE_COL]).copy()
    out = out[out[VALUE_COL] > 0].copy()

    if out.empty:
        raise ValueError(f"No positive seascape species-use rows found for {species}")

    out["month"] = out["date"].dt.month
    grouped = out.groupby(["month", "h3"], as_index=False)[VALUE_COL]

    if agg == "non_zero_median":
        monthly_raw = grouped.median()
    else:
        monthly_raw = grouped.mean()

    return pd.DataFrame(
        {
            "month": monthly_raw["month"].to_numpy(),
            "h3": monthly_raw["h3"].to_numpy(),
            agg: monthly_raw[VALUE_COL].to_numpy(),
        }
    )


def shared_norm(values: pd.Series, vmax_quantile: float) -> colors.Normalize:
    """Return a shared color scale for all monthly panels."""
    positive = values[values > 0].dropna()

    if positive.empty:
        raise ValueError("Color scale requires positive values")

    vmin = 0.0
    vmax = float(positive.quantile(vmax_quantile))

    if vmax <= vmin:
        vmax = float(positive.max())
    if vmax <= vmin:
        vmax = 1.0

    return colors.Normalize(vmin=vmin, vmax=vmax)


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    value_col: str,
    month: int,
    norm: colors.Normalize,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly species-use panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask].drop(columns=["month"])
    plot_gdf = grid.merge(month_values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=[value_col])

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            column=value_col,
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


def plot_monthly_matrix(
    monthly: pd.DataFrame,
    species: str,
    year: int,
    model_name: str,
    agg: str,
    vmax_quantile: float,
) -> Path:
    """Plot a 3-by-4 monthly species-use matrix."""
    value_col = normalize_agg(agg)
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = shared_norm(
        cast(pd.Series, monthly[value_col]),
        vmax_quantile=vmax_quantile,
    )

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
            monthly=monthly,
            value_col=value_col,
            month=month,
            norm=norm,
            bounds=bounds,
            land=land,
            coast=coast,
        )

    fig.suptitle(
        f"Monthly Seascape-Conditioned Species Use — {species}, {year}",
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
    cbar.set_label("Residence index")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)

    out_file = (
        OUTPUT_ROOT
        / f"seascape_species_use_{value_col}_{model_name}_{species}_{year}_all_months.png"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create monthly maps of seascape-conditioned species use.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--species",
        nargs="+",
        default=SPECIES,
        help="Species codes to plot.",
    )
    parser.add_argument(
        "--agg",
        default=AGG,
        choices=("non_zero_median", "non_zero_mean", "non_zero_media"),
    )
    parser.add_argument(
        "--vmax-quantile",
        type=float,
        default=0.99,
        help="Upper quantile for the shared color scale.",
    )

    return parser.parse_args()


def main() -> int:
    """Run seascape-conditioned species-use monthly mapping."""
    args = parse_args()
    df = load_surface(year=args.year, model_name=args.model_name)

    for species in args.species:
        monthly = summarize_monthly_species_use(
            df,
            species=species,
            agg=args.agg,
        )
        out_file = plot_monthly_matrix(
            monthly=monthly,
            species=species,
            year=args.year,
            model_name=args.model_name,
            agg=args.agg,
            vmax_quantile=args.vmax_quantile,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
