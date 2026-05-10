"""Plot yearly environmental plausibility summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths
from riskscape.visualization.maps import plausibility_path


MODEL_NAME = "bayesian_gmm"
PRODUCT_NAME = "joint"
YEARS = list(range(2014, 2024))
OUTPUT_ROOT = paths["plots"] / "plausibility"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "plausibility"

SPECIES_COLORS = {
    "BBAL": "#4c78a8",
    "SAFS": "#59a14f",
}


def parse_years(value: str) -> list[int]:
    """Parse all, a range, or comma-separated years."""
    if value.lower() == "all":
        return YEARS

    years: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            years.update(range(int(start_text), int(end_text) + 1))
        else:
            years.add(int(item))

    if not years:
        raise ValueError("No years selected")

    return sorted(years)


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"

    return "_".join(str(year) for year in years)


def summarize_year(
    year: int,
    model_name: str,
    product_name: str,
) -> pd.DataFrame:
    """Return yearly plausibility summaries by species."""
    path = plausibility_path(
        year=year,
        model_name=model_name,
        product_name=product_name,
    )

    if not path.exists():
        raise FileNotFoundError(f"Plausibility partition not found: {path}")

    query = """
        SELECT
            species,
            avg(plausibility) AS mean_plausibility,
            avg(CASE WHEN plausibility > 0 THEN plausibility ELSE NULL END)
                AS non_zero_mean_plausibility,
            median(CASE WHEN plausibility > 0 THEN plausibility ELSE NULL END)
                AS non_zero_median_plausibility,
            count(*) AS cell_days,
            sum(CASE WHEN plausibility > 0 THEN 1 ELSE 0 END)
                AS non_zero_cell_days
        FROM read_parquet(?)
        GROUP BY species
        ORDER BY species
    """

    with duckdb.connect(database=":memory:") as con:
        out = con.execute(query, [str(path)]).df()

    out.insert(0, "year", year)

    return out


def summarize_years(
    years: list[int],
    model_name: str,
    product_name: str,
) -> pd.DataFrame:
    """Return yearly plausibility summaries for selected years."""
    frames = [
        summarize_year(
            year=year,
            model_name=model_name,
            product_name=product_name,
        )
        for year in years
    ]

    return pd.concat(frames, ignore_index=True)


def save_plot(df: pd.DataFrame, years: list[int], out_file: Path) -> None:
    """Save yearly non-zero median plausibility time series."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for species, species_df in df.groupby("species"):
        species_df = species_df.sort_values("year")
        ax.plot(
            species_df["year"],
            species_df["non_zero_median_plausibility"],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=SPECIES_COLORS.get(species, "#4c78a8"),
            label=species,
        )

    ax.set_title(
        f"Yearly Non-Zero Median Environmental Plausibility — {year_label(years)}",
        fontsize=10,
    )
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Non-zero median plausibility", fontsize=8)
    ax.set_xticks(years)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot yearly environmental plausibility summaries.",
    )
    parser.add_argument(
        "--years",
        default="all",
        help="Use 'all', one year, a range like 2014-2023, or a comma list.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Plausibility model name.",
    )
    parser.add_argument(
        "--product",
        default=PRODUCT_NAME,
        help="Plausibility product name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated plausibility figures.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated plausibility CSV exports.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the yearly plausibility plotting workflow."""
    args = parse_args()
    years = parse_years(args.years)
    label = year_label(years)

    df = summarize_years(
        years=years,
        model_name=args.model,
        product_name=args.product,
    )

    csv_file = args.data_output_root / f"yearly_plausibility_{label}.csv"
    png_file = args.output_root / f"yearly_non_zero_median_plausibility_{label}.png"

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)
    save_plot(df, years, png_file)

    print("Saved:", csv_file)
    print("Saved:", png_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
