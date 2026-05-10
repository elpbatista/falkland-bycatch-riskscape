"""Plot fishing activity maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map

YEAR = "all"
VALUE_COL = "fishing_activity"
AGG = "mean"
INPUT_ROOT = paths["data"] / "modeling" / "fishing_training"
OUTPUT_ROOT = paths["plots"] / "fishing_activity"


def fishing_activity_path(
    year: int,
) -> Path:
    """Return the fishing activity input partition path."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def available_years() -> list[int]:
    """Return available fishing-training years."""
    years: list[int] = []

    for year_dir in sorted(INPUT_ROOT.glob("year=*")):
        years.append(int(year_dir.name.split("=", maxsplit=1)[1]))

    return years


def years_to_load(year: str) -> list[int]:
    """Return years requested by the CLI year argument."""
    if year.lower() == "all":
        years = available_years()
        if not years:
            raise FileNotFoundError(f"No fishing activity partitions found: {INPUT_ROOT}")
        return years

    return [int(year)]


def load_fishing_activity(
    year: str = YEAR,
) -> pd.DataFrame:
    """Load fishing activity rows for one or all years."""
    frames: list[pd.DataFrame] = []

    for selected_year in years_to_load(year):
        path = fishing_activity_path(year=selected_year)

        if not path.exists():
            raise FileNotFoundError(f"Fishing activity file not found: {path}")

        frames.append(pd.read_parquet(path))

    return pd.concat(frames, ignore_index=True)


def summarize_fishing_activity(
    df: pd.DataFrame,
    value_col: str = VALUE_COL,
    agg: str = AGG,
) -> pd.DataFrame:
    """Summarize fishing activity by H3 cell."""
    if value_col not in df.columns:
        raise ValueError(f"Missing fishing activity column: {value_col}")

    out = df.dropna(subset=["h3", value_col]).copy()

    if out.empty:
        raise ValueError("No fishing activity rows found")

    if agg == "mean":
        out = out[out[value_col] > 0].copy()

    if out.empty:
        raise ValueError(f"No positive fishing activity rows found for {value_col}")

    value_name = f"{value_col}_{agg}"
    grouped = out.groupby("h3")[value_col]

    if agg == "mean":
        values = grouped.mean()
    elif agg == "median":
        values = grouped.median()
    elif agg == "max":
        values = grouped.max()
    elif agg == "sum":
        values = grouped.sum()
    else:
        raise ValueError("--agg must be one of: mean, median, max, sum")

    return pd.DataFrame(
        {
            "h3": values.index.to_numpy(),
            value_name: values.to_numpy(),
        }
    )


def plot_fishing_activity_map(
    df: pd.DataFrame,
    year_label: str,
    value_col: str = VALUE_COL,
    agg: str = AGG,
) -> Path:
    """Plot summarized fishing activity."""
    grid = load_grid(uint64=True)
    summary = summarize_fishing_activity(df, value_col=value_col, agg=agg)
    value_name = f"{value_col}_{agg}"
    gdf = grid.merge(summary, on="h3", how="left")

    out_file = OUTPUT_ROOT / f"{value_col}_{agg}_{year_label}.png"
    title = f"Fishing Activity — {year_label}"

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=title,
        out_file=out_file,
        style=MapStyle(
            color_quantile=None,
            color_scale="log",
            colorbar_title="mean vessel-hours",
            alpha_scale=False,
            show_reference_map=False,
        ),
    )


def year_label(year: str) -> str:
    """Return display-safe year text."""
    if year.lower() != "all":
        return year

    years = available_years()
    if not years:
        return "all_years"

    return f"{min(years)}-{max(years)}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot fishing activity summarized over one year or all years."
    )
    parser.add_argument(
        "--year",
        default=YEAR,
        help="Year to plot, or 'all' for all available years.",
    )
    parser.add_argument(
        "--agg",
        default=AGG,
        choices=("mean", "median", "max", "sum"),
        help="H3 aggregation method.",
    )
    return parser.parse_args()


def main() -> int:
    """Run fishing activity map plots."""
    args = parse_args()
    df = load_fishing_activity(year=args.year)
    out_file = plot_fishing_activity_map(
        df,
        year_label=year_label(args.year),
        agg=args.agg,
    )
    print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
