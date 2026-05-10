"""Plot 12-panel monthly matrices for dynamic environmental features."""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
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
AGG = "mean"
INPUT_ROOT = paths["data"] / "features" / "environmental"
OUTPUT_ROOT = paths["plots"] / "environmental_monthly_matrices"


@dataclass(frozen=True)
class VariableSpec:
    """Display and scaling settings for one environmental feature."""

    column: str
    label: str
    colorbar: str
    cmap: str
    center_zero: bool = False
    quantile: float = 0.99


# Comment variables in or out here to choose which seasonal matrices to make.
# The True sets center_zero=True for that variable.
# So values below zero and above zero are shown symmetrically around 0, which is useful for anomalies because negative and positive departures are both meaningful.
VARIABLE_SPECS = [
    # VariableSpec("sst", "SST", "SST (K)", "turbo"),
    # VariableSpec("ssh", "SSH", "SSH (m)", "viridis"),
    # VariableSpec("wind_speed", "Wind Speed", "Wind speed (m/s)", "magma"),
    VariableSpec("chl_log", "Log-CHL", "Log-CHL", "YlGn"),
    # VariableSpec("sst_anom", "SST Anomaly", "SST anomaly (K)", "RdBu_r", True),
    # VariableSpec("ssh_anom", "SSH Anomaly", "SSH anomaly (m)", "RdBu_r", True),
    # VariableSpec("wind_speed_anom", "Wind-Speed Anomaly", "Wind-speed anomaly (m/s)", "RdBu_r", True),
    # VariableSpec("chl_log_anom", "Log-CHL Anomaly", "Log-CHL anomaly", "RdBu_r", True),
    # VariableSpec("sst_grad", "SST Gradient", "SST local gradient", "magma"),
    # VariableSpec("ssh_grad", "SSH Gradient", "SSH local gradient", "viridis"),
    # VariableSpec("chl_log_grad", "Log-CHL Gradient", "Log-CHL local gradient", "YlGn"),
]


def environmental_path(year: int) -> Path:
    """Return the environmental feature partition path for a year."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def selected_specs(variable_names: str | None) -> list[VariableSpec]:
    """Return variable specs after applying an optional CLI filter."""
    if variable_names is None:
        return VARIABLE_SPECS

    requested = {
        name.strip()
        for name in variable_names.split(",")
        if name.strip()
    }
    specs = [spec for spec in VARIABLE_SPECS if spec.column in requested]
    missing = requested - {spec.column for spec in specs}

    if missing:
        raise ValueError(
            "Unknown or commented-out variables requested: "
            + ", ".join(sorted(missing))
        )

    return specs


def load_environmental_features(
    year: int,
    specs: list[VariableSpec],
) -> pd.DataFrame:
    """Load selected environmental feature columns for a year."""
    path = environmental_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Environmental feature file not found: {path}")

    columns = ["h3", "date", *[spec.column for spec in specs]]
    return pd.read_parquet(path, columns=columns)


def summarize_monthly_features(
    df: pd.DataFrame,
    specs: list[VariableSpec],
    agg: str,
) -> pd.DataFrame:
    """Summarize environmental features by month and H3 cell."""
    if agg not in {"mean", "median", "min", "max"}:
        raise ValueError("--agg must be one of: mean, median, min, max")

    columns = [spec.column for spec in specs]
    out = df.dropna(subset=["h3", "date"]).copy()
    out["month"] = out["date"].dt.month

    grouped = out.groupby(["month", "h3"], as_index=False)[columns]
    if agg == "mean":
        monthly = grouped.mean()
    elif agg == "median":
        monthly = grouped.median()
    elif agg == "min":
        monthly = grouped.min()
    else:
        monthly = grouped.max()

    return pd.DataFrame(monthly)


def feature_norm(
    values: pd.Series,
    spec: VariableSpec,
) -> colors.Normalize:
    """Return a shared color scale for all monthly panels."""
    clean = values.dropna()

    if clean.empty:
        raise ValueError(f"No values found for {spec.column}")

    if spec.center_zero:
        limit = float(clean.abs().quantile(spec.quantile))
        if limit <= 0:
            limit = float(clean.abs().max())
        if limit <= 0:
            limit = 1.0
        return colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    vmin = float(clean.quantile(1.0 - spec.quantile))
    vmax = float(clean.quantile(spec.quantile))

    if vmax <= vmin:
        vmin = float(clean.min())
        vmax = float(clean.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    return colors.Normalize(vmin=vmin, vmax=vmax)


def month_panel_title(month: int) -> str:
    """Return a compact month title."""
    return calendar.month_abbr[month]


def plot_month_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    spec: VariableSpec,
    month: int,
    norm: colors.Normalize,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly environmental feature panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask, ["h3", spec.column]]
    plot_gdf = grid.merge(month_values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=[spec.column])

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            column=spec.column,
            cmap=spec.cmap,
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


def output_path(spec: VariableSpec, year: int, agg: str) -> Path:
    """Return the output file path for one variable matrix."""
    return OUTPUT_ROOT / f"{spec.column}_{agg}_monthly_matrix_{year}.png"


def plot_variable_matrix(
    monthly: pd.DataFrame,
    spec: VariableSpec,
    year: int,
    agg: str,
) -> Path:
    """Plot a 3-by-4 monthly matrix for one environmental variable."""
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = feature_norm(cast(pd.Series, monthly[spec.column]), spec=spec)

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
            spec=spec,
            month=month,
            norm=norm,
            bounds=bounds,
            land=land,
            coast=coast,
        )

    fig.suptitle(f"Monthly {spec.label} - {year}", fontsize=16, y=0.985)
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
        ScalarMappable(norm=norm, cmap=plt.get_cmap(spec.cmap)),
        cax=cax,
    )
    cbar.set_label(spec.colorbar)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)

    out_file = output_path(spec, year=year, agg=agg)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create 3-by-4 monthly matrices for environmental features."
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument(
        "--agg",
        default=AGG,
        choices=("mean", "median", "min", "max"),
        help="Monthly H3 aggregation method.",
    )
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Optional comma-separated feature columns to plot. "
            "By default, the uncommented VARIABLE_SPECS entries are used."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run environmental monthly matrix plotting."""
    args = parse_args()
    specs = selected_specs(args.variables)
    df = load_environmental_features(year=args.year, specs=specs)
    monthly = summarize_monthly_features(df, specs=specs, agg=args.agg)

    for spec in specs:
        out_file = plot_variable_matrix(
            monthly=monthly,
            spec=spec,
            year=args.year,
            agg=args.agg,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
