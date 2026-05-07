"""Plot fishing activity maps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map

YEAR = 2022
VALUE_COL = "fishing_activity"
AGG = "mean"
INPUT_ROOT = paths["data"] / "modeling" / "fishing_training"
OUTPUT_ROOT = paths["plots"] / "fishing_activity"


def fishing_activity_path(
    year: int = YEAR,
) -> Path:
    """Return the fishing activity input partition path."""
    return INPUT_ROOT / f"year={year}" / "part.parquet"


def load_fishing_activity(
    year: int = YEAR,
) -> pd.DataFrame:
    """Load fishing activity rows for one year."""
    path = fishing_activity_path(
        year=year,
    )

    if not path.exists():
        raise FileNotFoundError(f"Fishing activity file not found: {path}")

    return pd.read_parquet(path)


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

    return (
        out.groupby("h3", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: value_name})
    )


def plot_fishing_activity_map(
    df: pd.DataFrame,
    year: int = YEAR,
    value_col: str = VALUE_COL,
    agg: str = AGG,
) -> Path:
    """Plot summarized fishing activity for one year."""
    grid = load_grid(uint64=True)
    summary = summarize_fishing_activity(df, value_col=value_col, agg=agg)
    value_name = f"{value_col}_{agg}"
    gdf = grid.merge(summary, on="h3", how="left")

    out_file = OUTPUT_ROOT / f"{value_col}_{agg}_{year}.png"
    title = f"Fishing Activity ({agg}) - {year}"

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=title,
        out_file=out_file,
        style=MapStyle(
            color_quantile=None,
            color_scale="log",
            alpha_scale=False,
            show_reference_map=False,
        ),
    )


def main() -> int:
    """Run fishing activity map plots."""
    df = load_fishing_activity()
    out_file = plot_fishing_activity_map(df)
    print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
