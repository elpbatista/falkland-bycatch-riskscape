"""Plot environmental gradient maps for SST, SSH, and log-chlorophyll."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map


DEFAULT_YEAR = 2022
DEFAULT_DATE = "2022-12-10"
DEFAULT_AGG = "mean"
INPUT_ROOT = paths["data"] / "features" / "environmental"
OUTPUT_ROOT = paths["plots"] / "environmental_gradients"

GRADIENT_LAYERS = {
    "sst_grad": {
        "label": "SST Gradient",
        "colorbar": "SST local gradient",
        "cmap": "magma",
    },
    "ssh_grad": {
        "label": "SSH Gradient",
        "colorbar": "SSH local gradient",
        "cmap": "viridis",
    },
    "chl_log_grad": {
        "label": "Log-CHL Gradient",
        "colorbar": "Log-CHL local gradient",
        "cmap": "YlGn",
    },
}


def environmental_path(year: int) -> Path:
    """Return the environmental feature partition path for a year."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def load_environmental_gradients(
    year: int,
    date: str | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Load environmental gradient columns for a selected time window."""
    path = environmental_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Environmental feature file not found: {path}")

    columns = ["h3", "date", *GRADIENT_LAYERS.keys()]

    if date is not None and month is not None:
        raise ValueError("Use either --date or --month, not both")

    if date is not None:
        target = pd.Timestamp(date, tz="UTC")
        out = pd.read_parquet(
            path,
            columns=columns,
            filters=[("date", "=", target)],
        )
        if out.empty:
            raise ValueError(f"No environmental rows found for date: {date}")
        return out

    out = pd.read_parquet(path, columns=columns)

    if month is not None:
        if month < 1 or month > 12:
            raise ValueError("--month must be between 1 and 12")
        out = out[out["date"].dt.month == month].copy()
        if out.empty:
            raise ValueError(f"No environmental rows found for month: {month}")

    return out


def summarize_gradients(
    df: pd.DataFrame,
    agg: str,
) -> pd.DataFrame:
    """Summarize gradient values by H3 cell."""
    if agg not in {"mean", "median", "max"}:
        raise ValueError("--agg must be one of: mean, median, max")

    return (
        df.groupby("h3", as_index=False)[list(GRADIENT_LAYERS)]
        .agg(agg)
        .dropna(how="all", subset=list(GRADIENT_LAYERS))
    )


def output_suffix(
    year: int,
    date: str | None,
    month: int | None,
    agg: str,
) -> str:
    """Return a stable filename suffix for the selected time window."""
    if date is not None:
        return date.replace("-", "")
    if month is not None:
        return f"{year}_month_{month:02d}_{agg}"
    return f"{year}_{agg}"


def title_suffix(
    year: int,
    date: str | None,
    month: int | None,
    agg: str,
) -> str:
    """Return display text for the selected time window."""
    if date is not None:
        return date
    if month is not None:
        return f"{agg.title()} gradient, {year} month {month:02d}"
    return f"{agg.title()} gradient, {year}"


def plot_gradient_maps(
    df: pd.DataFrame,
    year: int,
    date: str | None,
    month: int | None,
    agg: str,
) -> list[Path]:
    """Plot SST, SSH, and log-chlorophyll gradient maps."""
    grid = load_grid(uint64=True)
    summary = summarize_gradients(df, agg=agg)
    gdf = grid.merge(summary, on="h3", how="left")

    suffix = output_suffix(year, date, month, agg)
    label_suffix = title_suffix(year, date, month, agg)
    out_files: list[Path] = []

    for column, meta in GRADIENT_LAYERS.items():
        out_file = OUTPUT_ROOT / f"{column}_{suffix}.png"
        title = f"{meta['label']} - {label_suffix}"
        out_files.append(
            plot_h3_map(
                gdf=gdf,
                value_col=column,
                title=title,
                out_file=out_file,
                style=MapStyle(
                    cmap=meta["cmap"],
                    colorbar_title=meta["colorbar"],
                    color_quantile=0.99,
                    min_display_value=0.0,
                    alpha_scale=True,
                    alpha=0.9,
                    alpha_min=0.25,
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
        description=(
            "Create H3 maps of SST, SSH, and log-CHL gradient layers. "
            "The maps show gradient layers without a bathymetry base layer."
        )
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Representative date to plot as YYYY-MM-DD. Use --date none for a year/month aggregate.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional month to aggregate when --date none is used.",
    )
    parser.add_argument(
        "--agg",
        default=DEFAULT_AGG,
        choices=("mean", "median", "max"),
        help="H3 aggregation for month/year maps.",
    )
    return parser.parse_args()


def main() -> int:
    """Run environmental gradient map plotting."""
    args = parse_args()
    date = None if str(args.date).lower() in {"none", "null", ""} else args.date

    df = load_environmental_gradients(
        year=args.year,
        date=date,
        month=args.month,
    )
    out_files = plot_gradient_maps(
        df=df,
        year=args.year,
        date=date,
        month=args.month,
        agg=args.agg,
    )

    for out_file in out_files:
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
