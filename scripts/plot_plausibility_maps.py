"""Plot Bayesian GMM environmental plausibility maps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.maps import MapStyle, plot_h3_map


YEAR = 2022
MODEL_NAME = "bayesian_gmm"
VALUE_COL = "plausibility"
AGG = "mean"

INPUT_ROOT = paths["data"] / "modeling" / "plausibility" / MODEL_NAME
OUTPUT_ROOT = paths["plots"] / "plausibility" 


def plausibility_path(product_name: str, year: int = YEAR) -> Path:
    """Return the plausibility input partition path."""
    return INPUT_ROOT / product_name / f"year={year}" / "part.parquet"


def load_plausibility(product_name: str, year: int = YEAR) -> pd.DataFrame:
    """Load plausibility rows for one product and year."""
    path = plausibility_path(product_name=product_name, year=year)

    if not path.exists():
        raise FileNotFoundError(f"Plausibility file not found: {path}")

    return pd.read_parquet(path, columns=["h3", "species", "plausibility"])


def summarize_plausibility(
    df: pd.DataFrame,
    species: str,
    value_col: str = VALUE_COL,
    agg: str = AGG,
) -> pd.DataFrame:
    """Summarize plausibility by H3 cell, keeping zero rows in the mean."""
    if value_col not in df.columns:
        raise ValueError(f"Missing plausibility column: {value_col}")

    out = df[df["species"] == species].dropna(subset=["h3", value_col]).copy()

    if out.empty:
        raise ValueError(f"No plausibility rows found for {species}")

    value_name = f"{value_col}_{agg}"

    return (
        out.groupby("h3", as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: value_name})
    )


def plot_plausibility_map(
    df: pd.DataFrame,
    product_name: str,
    species: str,
    year: int = YEAR,
    value_col: str = VALUE_COL,
    agg: str = AGG,
) -> Path:
    """Plot mean plausibility for one product/species/year."""
    grid = load_grid(uint64=True)
    summary = summarize_plausibility(
        df=df,
        species=species,
        value_col=value_col,
        agg=agg,
    )
    value_name = f"{value_col}_{agg}"
    gdf = grid.merge(summary, on="h3", how="left")

    out_file = (
        OUTPUT_ROOT
        / f"{MODEL_NAME}_{product_name}_{value_col}_{agg}_{species}_{year}.png"
    )
    title = f"Plausibility ({agg}) - {MODEL_NAME}/{product_name} - {species} - {year}"

    return plot_h3_map(
        gdf=gdf,
        value_col=value_name,
        title=title,
        out_file=out_file,
        style=MapStyle(
            cmap="viridis",
            color_quantile=0.99,
            show_reference_map=False,
        ),
    )


def main() -> int:
    """Run plausibility map plots."""
    plot_products = [
        ("bbal", "BBAL"),
        ("safs", "SAFS"),
        ("joint", "BBAL"),
        ("joint", "SAFS"),
    ]

    for product_name, species in plot_products:
        print(
            "Input: "
            f"data/modeling/plausibility/{MODEL_NAME}/{product_name}/"
            f"year={YEAR}/part.parquet"
        )
        df = load_plausibility(product_name=product_name, year=YEAR)
        out_file = plot_plausibility_map(
            df=df,
            product_name=product_name,
            species=species,
            year=YEAR,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
