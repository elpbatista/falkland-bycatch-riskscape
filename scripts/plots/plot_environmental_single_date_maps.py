"""Plot single-date maps for environmental feature layers."""

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

import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.utils.dates import read_parquet_utc_day
from riskscape.visualization.maps import MapStyle, plot_h3_map


YEAR = 2022
DATE = "2022-12-10"
INPUT_ROOT = paths["data"] / "features" / "environmental"
OUTPUT_ROOT = paths["plots"] / "environmental_single_date"


@dataclass(frozen=True)
class VariableSpec:
    """Display and scaling settings for one environmental feature."""

    column: str
    label: str
    colorbar: str
    cmap: str
    center_zero: bool = False
    quantile: float = 0.99


# Comment variables in or out here to choose which single-date maps to make.
# Set center_zero=True for anomaly layers so positive and negative departures
# are shown symmetrically around zero.
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
    # wind_speed_grad is not currently generated in data/features/environmental.
    # VariableSpec("wind_speed_grad", "Wind Speed Gradient", "Wind speed gradient", "magma"),
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


def load_environmental_date(
    year: int,
    date: str,
    specs: list[VariableSpec],
) -> pd.DataFrame:
    """Load selected environmental feature columns for one date."""
    path = environmental_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Environmental feature file not found: {path}")

    columns = ["h3", "date", *[spec.column for spec in specs]]
    out = read_parquet_utc_day(path, columns=columns, date=date)

    if out.empty:
        raise ValueError(f"No environmental rows found for date: {date}")

    return out


def feature_limits(
    values: pd.Series,
    spec: VariableSpec,
) -> tuple[float, float]:
    """Return color limits for a feature map."""
    clean = values.dropna()

    if clean.empty:
        raise ValueError(f"No values found for {spec.column}")

    if spec.center_zero:
        limit = float(clean.abs().quantile(spec.quantile))
        if limit <= 0:
            limit = float(clean.abs().max())
        if limit <= 0:
            limit = 1.0
        return -limit, limit

    color_min = float(clean.quantile(1.0 - spec.quantile))
    color_max = float(clean.quantile(spec.quantile))

    if color_max <= color_min:
        color_min = float(clean.min())
        color_max = float(clean.max())
    if color_max <= color_min:
        color_max = color_min + 1.0

    return color_min, color_max


def display_value(value: float, spec: VariableSpec) -> float:
    """Return the value to display on the colorbar endpoint label."""
    if "chl_log" in spec.column:
        return float(np.expm1(value))

    if spec.column == "sst":
        return value - 273.15

    return value


def endpoint_label(value: float, spec: VariableSpec) -> str:
    """Return compact endpoint label text for a colorbar."""
    shown = display_value(value, spec)
    abs_value = abs(shown)

    if abs_value >= 100:
        return f"{shown:.0f}"
    if abs_value >= 10:
        return f"{shown:.1f}"
    if abs_value >= 1:
        return f"{shown:.2f}"
    return f"{shown:.3f}"


def date_suffix(date: str) -> str:
    """Return compact date text for filenames."""
    return date.replace("-", "")


def plot_single_date_maps(
    df: pd.DataFrame,
    date: str,
    specs: list[VariableSpec],
) -> list[Path]:
    """Plot one H3 map per selected environmental variable."""
    grid = load_grid(uint64=True)
    gdf = grid.merge(df.drop(columns=["date"]), on="h3", how="left")
    out_files: list[Path] = []

    for spec in specs:
        color_min, color_max = feature_limits(cast(pd.Series, gdf[spec.column]), spec)
        out_file = OUTPUT_ROOT / f"{spec.column}_{date_suffix(date)}.png"
        out_files.append(
            plot_h3_map(
                gdf=gdf,
                value_col=spec.column,
                title=f"{spec.label} — {date}",
                out_file=out_file,
                style=MapStyle(
                    legend_mode=(
                        "diverging_centered" if spec.center_zero else "continuous"
                    ),
                    cmap=spec.cmap,
                    colorbar_title=spec.colorbar,
                    color_min=color_min,
                    color_max=color_max,
                    color_quantile=None,
                    colorbar_bottom_label=endpoint_label(color_min, spec),
                    colorbar_top_label=endpoint_label(color_max, spec),
                    min_display_value=None,
                    alpha_scale=False,
                    bathymetry=False,
                    bathymetry_log_scale=False,
                    show_reference_map=False,
                ),
            )
        )

    return out_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create single-date H3 maps for environmental feature layers."
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--date", default=DATE, help="Date to plot as YYYY-MM-DD.")
    parser.add_argument(
        "--variables",
        default=None,
        help=(
            "Optional comma-separated feature columns. "
            "By default, uncommented VARIABLE_SPECS entries are used."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run single-date environmental map plotting."""
    args = parse_args()
    specs = selected_specs(args.variables)
    df = load_environmental_date(
        year=args.year,
        date=args.date,
        specs=specs,
    )
    out_files = plot_single_date_maps(
        df=df,
        date=args.date,
        specs=specs,
    )

    for out_file in out_files:
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
