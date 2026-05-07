"""Plot species presence feature maps across all available years."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map


FEATURE_ROOT = paths["data"] / "features" / "species_presence"
OUTPUT_ROOT = paths["plots"] / "species_presence"


def load_species_presence() -> pd.DataFrame:
    """Load all species presence feature partitions."""
    parts = sorted(FEATURE_ROOT.glob("year=*/part.parquet"))

    if not parts:
        raise FileNotFoundError(f"No species presence partitions found: {FEATURE_ROOT}")

    return pd.concat((pd.read_parquet(path) for path in parts), ignore_index=True)


def summarize_presence(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Summarize one species across all years by H3 cell."""
    out = df[df["species"] == species].copy()

    if out.empty:
        raise ValueError(f"No species presence rows found for {species}")

    return (
        out.groupby("h3", as_index=False)["presence_count"]
        .sum()
        .rename(columns={"presence_count": "presence_count_total"})
    )


def plot_species_presence_map(df: pd.DataFrame, species: str) -> Path:
    """Plot all-year presence for one species."""
    grid = load_grid(uint64=True)
    summary = summarize_presence(df, species)
    gdf = grid.merge(summary, on="h3", how="left")

    out_file = OUTPUT_ROOT / f"{species.lower()}_presence_count_all_years.png"
    title = f"{species} Presence Count - All Years"

    return plot_h3_map(
        gdf=gdf,
        value_col="presence_count_total",
        title=title,
        out_file=out_file,
        style=MapStyle(
            cmap="inferno",
            color_scale="log",
            color_quantile=0.999,
            alpha_scale=False,
            show_reference_map=False,
        ),
    )


def main() -> int:
    """Plot one all-year species presence map per species."""
    df = load_species_presence()

    for species in sorted(df["species"].dropna().unique()):
        out_file = plot_species_presence_map(df, species)
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
