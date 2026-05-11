"""Plot fishing-hours totals and unique vessel counts over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths


YEARS = "2014-2023"
INPUT_ROOT = paths["data"] / "raw" / "gfw"
OUTPUT_ROOT = paths["plots"] / "fishing_activity"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "fishing_activity"


def fishing_effort_path(year: int) -> Path:
    """Return the raw fishing-effort partition path for one year."""
    return INPUT_ROOT / f"year={year}" / "fishing_effort.parquet"


def available_years() -> list[int]:
    """Return available raw fishing-effort years."""
    years: list[int] = []

    for year_dir in sorted(INPUT_ROOT.glob("year=*")):
        years.append(int(year_dir.name.split("=", maxsplit=1)[1]))

    if not years:
        raise FileNotFoundError(f"No raw fishing-effort partitions found: {INPUT_ROOT}")

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


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if len(years) == 1:
        return str(years[0])
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"
    return "_".join(str(year) for year in years)


def load_year_activity(year: int) -> pd.DataFrame:
    """Load one raw fishing-effort year."""
    path = fishing_effort_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Raw fishing-effort file not found: {path}")

    df = pd.read_parquet(
        path,
        columns=["date", "hours", "vessel_id"],
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].dt.tz_convert(None)

    return df


def summarize_period(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Summarize fishing hours and unique vessels for a date column."""
    return (
        df.groupby(date_col, as_index=False)
        .agg(
            fishing_hours=("hours", "sum"),
            vessel_count=("vessel_id", "nunique"),
        )
        .rename(columns={date_col: "date"})
        .sort_values("date")
    )


def summarize_year_activity(year: int, daily: bool) -> pd.DataFrame:
    """Summarize one year as daily or monthly values."""
    df = load_year_activity(year)

    if daily:
        df["period"] = df["date"].dt.floor("D")
    else:
        df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return summarize_period(df, "period")


def compute_activity_totals(years: list[int]) -> tuple[pd.DataFrame, str]:
    """Compute daily values for one year or monthly values for multiple years."""
    daily = len(years) == 1
    frames = [summarize_year_activity(year, daily=daily) for year in years]
    activity = pd.concat(frames, ignore_index=True)

    activity = (
        activity.groupby("date", as_index=False)
        .agg(
            fishing_hours=("fishing_hours", "sum"),
            vessel_count=("vessel_count", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return activity, "daily" if daily else "monthly"


def save_activity_totals(
    activity: pd.DataFrame,
    years: list[int],
    time_step: str,
) -> Path:
    """Save fishing activity totals as CSV."""
    out_file = (
        DATA_OUTPUT_ROOT
        / f"fishing_activity_{time_step}_totals_{year_label(years)}.csv"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    activity.to_csv(out_file, index=False)

    return out_file


def plot_activity_totals(
    activity: pd.DataFrame,
    years: list[int],
    time_step: str,
) -> Path:
    """Plot fishing hours and unique vessel counts in stacked panels."""
    fig, (ax_hours, ax_vessels) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 0.5]},
    )

    ax_hours.plot(
        activity["date"],
        activity["fishing_hours"],
        color="#d95f02",
        linewidth=1.4,
        label="Fishing hours",
    )
    ax_vessels.plot(
        activity["date"],
        activity["vessel_count"],
        color="#1b9e77",
        linewidth=1.4,
        label="Vessel count",
    )

    title_step = "Daily" if time_step == "daily" else "Monthly"
    ax_hours.set_title(
        f"{title_step} Fishing Activity — {year_label(years)}",
        fontsize=11,
    )
    ax_hours.set_ylabel("Fishing hours", fontsize=8, color="#d95f02")
    ax_vessels.set_xlabel("Year", fontsize=8)
    ax_vessels.set_ylabel("Unique vessels", fontsize=8, color="#1b9e77")

    ax_hours.tick_params(axis="both", labelsize=7)
    ax_vessels.tick_params(axis="both", labelsize=7)
    if len(years) == 1:
        ax_vessels.xaxis.set_major_locator(mdates.MonthLocator())
        ax_vessels.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    else:
        ax_vessels.xaxis.set_major_locator(mdates.YearLocator())
        ax_vessels.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in (ax_hours, ax_vessels):
        ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.legend(frameon=False, fontsize=7, loc="upper left")

    for spine in ax_hours.spines.values():
        spine.set_visible(False)
    for spine in ax_vessels.spines.values():
        spine.set_visible(False)

    out_file = (
        OUTPUT_ROOT
        / f"fishing_activity_{time_step}_totals_{year_label(years)}.png"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(hspace=0.18)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot fishing-hour totals and unique vessel counts from "
            "the raw GFW fishing-effort table."
        )
    )
    parser.add_argument(
        "--years",
        default=YEARS,
        help="Years to plot: all, one year, a range like 2014-2023, or comma-separated years.",
    )

    return parser.parse_args()


def main() -> int:
    """Run monthly fishing activity time-series plotting."""
    args = parse_args()
    years = parse_years(args.years)
    activity, time_step = compute_activity_totals(years)
    csv_file = save_activity_totals(activity, years, time_step)
    png_file = plot_activity_totals(activity, years, time_step)

    print(f"Saved: {csv_file}")
    print(f"Saved: {png_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
