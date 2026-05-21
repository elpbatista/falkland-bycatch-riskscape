"""Plot 12-panel monthly matrices for dynamic environmental features."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
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
from riskscape.visualization.legends import label_colorbar_extremes
from riskscape.visualization.base_map import MapBounds, load_reference_layers
from riskscape.visualization.maps import draw_h3_column_panel
from riskscape.visualization.monthly_maps import (
    add_monthly_colorbar_axis,
    create_monthly_map_grid,
    format_month_panel,
    month_axes,
    save_monthly_map,
)

from plot_environmental_single_date_maps import (  # noqa: E402
    endpoint_label,
    feature_limits,
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


# Standard environmental monthly matrices. Use --variables to plot a subset.
VARIABLE_SPECS = [
    VariableSpec("sst", "SST", "SST (°C)", "turbo"),
    VariableSpec("ssh", "SSH", "SSH (m)", "viridis"),
    VariableSpec("wind_speed", "Wind Speed", "Wind speed (m/s)", "magma"),
    VariableSpec("chl_log", "CHL", "CHL (mg m^-3)", "YlGn"),
    VariableSpec("sst_anom", "SST Anomaly", "SST anomaly (°C)", "RdBu_r", True),
    VariableSpec("ssh_anom", "SSH Anomaly", "SSH anomaly (m)", "RdBu_r", True),
    VariableSpec(
        "wind_speed_anom",
        "Wind-Speed Anomaly",
        "Wind-speed anomaly (m/s)",
        "RdBu_r",
        True,
    ),
    VariableSpec(
        "chl_log_anom",
        "CHL Anomaly",
        "CHL anomaly (mg m^-3)",
        "RdBu_r",
        True,
    ),
    VariableSpec("sst_grad", "SST Gradient", "SST local gradient", "magma"),
    VariableSpec("ssh_grad", "SSH Gradient", "SSH local gradient", "viridis"),
    VariableSpec("chl_log_grad", "CHL Gradient", "CHL local gradient (mg m^-3)", "YlGn"),
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
    format_month_panel(ax, month=month, bounds=bounds)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask, ["h3", spec.column]]
    draw_h3_column_panel(
        ax=ax,
        grid=grid,
        values=month_values,
        value_col=spec.column,
        norm=norm,
        cmap=spec.cmap,
        bounds=bounds,
        land=land,
        coast=coast,
    )


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
    color_min, color_max = feature_limits(cast(pd.Series, monthly[spec.column]), spec)
    if spec.center_zero:
        norm = colors.TwoSlopeNorm(vmin=color_min, vcenter=0.0, vmax=color_max)
    else:
        norm = colors.Normalize(vmin=color_min, vmax=color_max)

    fig, axes_flat = create_monthly_map_grid(f"Monthly {spec.label} - {year}")

    for month, ax in month_axes(axes_flat):
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

    cax = add_monthly_colorbar_axis(fig)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(spec.cmap)),
        cax=cax,
    )
    cbar.set_label(spec.colorbar)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)
    label_colorbar_extremes(
        cbar,
        bottom=endpoint_label(float(norm.vmin), spec),
        top=endpoint_label(float(norm.vmax), spec),
    )

    out_file = output_path(spec, year=year, agg=agg)
    return save_monthly_map(fig, out_file)


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
