"""Plot daily mean time series for environmental feature layers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths


YEARS = "all"
INPUT_ROOT = paths["data"] / "features" / "environmental"
OUTPUT_ROOT = paths["plots"] / "environmental_timeseries"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "environmental_timeseries"

Transform = Literal["kelvin_to_c", "expm1"]


@dataclass(frozen=True)
class VariableSpec:
    """Display settings for one environmental feature."""

    column: str
    label: str
    ylabel: str
    color: str
    transform: Transform | None = None


# Comment variables in or out here to choose which daily mean time series to make.
VARIABLE_SPECS = [
    VariableSpec("sst", "SST", "SST (°C)", "#f46d43", "kelvin_to_c"),
    VariableSpec("ssh", "SSH", "SSH (m)", "#4575b4"),
    VariableSpec("wind_speed", "Wind Speed", "Wind speed (m/s)", "#7b3294"),
    VariableSpec("chl_log", "CHL", "CHL (mg m^-3)", "#66bd63", "expm1"),
    VariableSpec("sst_anom", "SST Anomaly", "SST anomaly (°C)", "#f46d43"),
    VariableSpec("ssh_anom", "SSH Anomaly", "SSH anomaly (m)", "#4575b4"),
    VariableSpec(
        "wind_speed_anom",
        "Wind-Speed Anomaly",
        "Wind-speed anomaly (m/s)",
        "#7b3294",
    ),
    VariableSpec("chl_log_anom", "CHL Anomaly", "CHL anomaly (log units)", "#66bd63"),
    VariableSpec("sst_grad", "SST Gradient", "SST local gradient (°C)", "#f46d43"),
    VariableSpec("ssh_grad", "SSH Gradient", "SSH local gradient", "#4575b4"),
    VariableSpec("chl_log_grad", "CHL Gradient", "CHL local gradient (log units)", "#66bd63"),
]


def environmental_path(year: int) -> Path:
    """Return the environmental feature partition path for a year."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def available_years() -> list[int]:
    """Return available environmental feature years."""
    years: list[int] = []

    for year_dir in sorted(INPUT_ROOT.glob("year=*")):
        years.append(int(year_dir.name.split("=", maxsplit=1)[1]))

    if not years:
        raise FileNotFoundError(f"No environmental partitions found: {INPUT_ROOT}")

    return years


def parse_years(years: str) -> list[int]:
    """Parse all, a single year, ranges, or comma-separated years."""
    if years.lower() == "all":
        return available_years()

    parsed: set[int] = set()
    for part in years.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(item))

    if not parsed:
        raise ValueError("No years selected")

    return sorted(parsed)


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


def load_year_daily_means(year: int, specs: list[VariableSpec]) -> pd.DataFrame:
    """Load one year and compute daily spatial means."""
    path = environmental_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Environmental feature file not found: {path}")

    columns = ["date", *[spec.column for spec in specs]]
    df = pd.read_parquet(path, columns=columns)

    return df.groupby("date", as_index=False).mean(numeric_only=True)


def compute_daily_means(
    years: list[int],
    specs: list[VariableSpec],
) -> pd.DataFrame:
    """Compute daily means across selected years."""
    frames = [load_year_daily_means(year, specs) for year in years]
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def apply_transform(values: pd.Series, spec: VariableSpec) -> pd.Series:
    """Transform values for display."""
    if spec.transform == "kelvin_to_c":
        return values - 273.15
    if spec.transform == "expm1":
        return pd.Series(
            np.expm1(values.to_numpy()),
            index=values.index,
            name=values.name,
        )

    return values


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if len(years) == 1:
        return str(years[0])
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"
    return "_".join(str(year) for year in years)


def plot_variable_timeseries(
    daily: pd.DataFrame,
    spec: VariableSpec,
    years: list[int],
) -> Path:
    """Plot one environmental daily mean time series."""
    values = apply_transform(daily[spec.column], spec)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["date"], values, color=spec.color, linewidth=1.0)
    ax.set_title(f"Daily Mean {spec.label} — {year_label(years)}", fontsize=11)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel(spec.ylabel, fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    if len(years) == 1:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out_file = OUTPUT_ROOT / f"{spec.column}_daily_mean_{year_label(years)}.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def save_daily_means(
    daily: pd.DataFrame,
    years: list[int],
) -> Path:
    """Save daily mean values as CSV."""
    out_file = DATA_OUTPUT_ROOT / f"environmental_daily_means_{year_label(years)}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_file, index=False)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot daily spatial mean time series for environmental features."
    )
    parser.add_argument(
        "--years",
        default=YEARS,
        help="Use 'all', one year, a range like 2018-2022, or a comma list.",
    )
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
    """Run environmental daily mean time-series plotting."""
    args = parse_args()
    specs = selected_specs(args.variables)
    years = parse_years(args.years)
    daily = compute_daily_means(years, specs)

    csv_file = save_daily_means(daily, years)
    print(f"Saved: {csv_file}")

    for spec in specs:
        out_file = plot_variable_timeseries(daily, spec, years)
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
