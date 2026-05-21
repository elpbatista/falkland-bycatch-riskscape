"""Plot a 12-panel monthly fishing-activity matrix."""

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

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import geopandas as gpd
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
VALUE_COL = "fishing_activity"
AGG = "non_zero_median"
INPUT_ROOT = paths["data"] / "modeling" / "fishing_training"
OUTPUT_ROOT = paths["plots"] / "fishing_activity"
CMAP = "YlOrRd"


def fishing_activity_path(year: int) -> Path:
    """Return the fishing activity input partition path."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def load_fishing_activity(year: int) -> pd.DataFrame:
    """Load fishing activity columns for one year."""
    path = fishing_activity_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Fishing activity file not found: {path}")

    return pd.read_parquet(path, columns=["h3", "date", VALUE_COL])


def normalize_agg(agg: str) -> str:
    """Normalize accepted aggregation names."""
    if agg == "non_zero_media":
        return "non_zero_median"
    return agg


def summarize_monthly_activity(
    df: pd.DataFrame,
    agg: str = AGG,
) -> pd.DataFrame:
    """Summarize non-zero fishing activity by month and H3 cell."""
    agg = normalize_agg(agg)
    if agg not in {"non_zero_median", "non_zero_mean"}:
        raise ValueError("--agg must be non_zero_median or non_zero_mean")

    out = df.dropna(subset=["h3", "date", VALUE_COL]).copy()
    out = out[out[VALUE_COL] > 0].copy()

    if out.empty:
        raise ValueError("No positive fishing activity rows found")

    out["month"] = out["date"].dt.month
    stat = "median" if agg == "non_zero_median" else "mean"

    grouped = out.groupby(["month", "h3"], as_index=False)[VALUE_COL]
    if stat == "median":
        monthly_raw = grouped.median()
    else:
        monthly_raw = grouped.mean()

    monthly = pd.DataFrame(
        {
            "month": monthly_raw["month"].to_numpy(),
            "h3": monthly_raw["h3"].to_numpy(),
            agg: monthly_raw[VALUE_COL].to_numpy(),
        }
    )

    return monthly


def shared_log_norm(values: pd.Series, vmax_quantile: float) -> colors.LogNorm:
    """Return a shared log color scale for all monthly panels."""
    positive = values[values > 0].dropna()

    if positive.empty:
        raise ValueError("Log color scale requires positive values")

    vmin = float(positive.min())
    vmax = float(positive.quantile(vmax_quantile))

    if vmax <= vmin:
        vmax = float(positive.max())
    if vmax <= vmin:
        vmax = vmin * 1.01

    return colors.LogNorm(vmin=vmin, vmax=vmax)


def month_panel_title(month: int) -> str:
    """Return a compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    value_col: str,
    month: int,
    norm: colors.LogNorm,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly fishing activity panel."""
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
    year: int,
    agg: str,
    vmax_quantile: float,
) -> Path:
    """Plot the 3-by-4 monthly fishing-activity matrix."""
    value_col = normalize_agg(agg)
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = shared_log_norm(
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
        f"Monthly Fishing Activity - {year}",
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
    cbar.set_label("mean vessel-hours")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)

    out_file = OUTPUT_ROOT / f"{VALUE_COL}_{value_col}_monthly_matrix_{year}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a 3-by-4 matrix of monthly fishing-activity maps for one year."
        )
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument(
        "--agg",
        default=AGG,
        help="Use non_zero_median or non_zero_mean. The typo non_zero_media is accepted as non_zero_median.",
    )
    parser.add_argument(
        "--vmax-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used for the shared log color scale.",
    )
    return parser.parse_args()


def main() -> int:
    """Run monthly fishing activity matrix plotting."""
    args = parse_args()
    agg = normalize_agg(args.agg)

    df = load_fishing_activity(year=args.year)
    monthly = summarize_monthly_activity(df, agg=agg)
    out_file = plot_monthly_matrix(
        monthly=monthly,
        year=args.year,
        agg=agg,
        vmax_quantile=args.vmax_quantile,
    )
    print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
