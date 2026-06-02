"""Plot temporal coverage of species telemetry-derived observations."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths


FEATURE_ROOT = paths["data"] / "features" / "species_presence"
OUTPUT_ROOT = paths["plots"] / "species_presence"
OUTPUT_FILE = OUTPUT_ROOT / "species_temporal_coverage_matrix.png"
SUMMARY_FILE = OUTPUT_ROOT / "species_temporal_coverage_summary.csv"

SPECIES_COLORS = {
    "BBAL": "#4c78a8",
    "SAFS": "#f58518",
}


def load_species_presence() -> pd.DataFrame:
    """Load all species presence feature partitions."""
    parts = sorted(FEATURE_ROOT.glob("year=*/part.parquet"))
    if not parts:
        raise FileNotFoundError(f"No species presence partitions found: {FEATURE_ROOT}")

    df = pd.concat(
        (
            pd.read_parquet(
                path,
                columns=[
                    "h3",
                    "date",
                    "species",
                    "presence_count",
                    "individual_count",
                    "trip_count",
                ],
            )
            for path in parts
        ),
        ignore_index=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize telemetry-derived species support by date and species."""
    out = (
        df.groupby(["species", "date"], as_index=False)
        .agg(
            h3_cells=("h3", "nunique"),
            presence_count=("presence_count", "sum"),
            individual_cell_count=("individual_count", "sum"),
            trip_cell_count=("trip_count", "sum"),
        )
        .sort_values(["species", "date"])
    )
    out["year"] = out["date"].dt.year
    return out


def species_totals(summary: pd.DataFrame) -> pd.DataFrame:
    """Return species-level temporal coverage totals for annotation/export."""
    return (
        summary.groupby("species", as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            observed_dates=("date", "nunique"),
            h3_cell_days=("h3_cells", "sum"),
            presence_count=("presence_count", "sum"),
        )
        .sort_values("species")
    )


def plot_temporal_coverage_matrix(summary: pd.DataFrame, out_file: Path) -> Path:
    """Plot species-by-year circular temporal coverage panels."""
    species_values = ["BBAL", "SAFS"]
    years = [2022, 2023]
    fig, axes = plt.subplots(
        nrows=len(species_values),
        ncols=len(years),
        figsize=(9.2, 8.6),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )

    species_max = (
        summary.groupby("species")["presence_count"].max().astype("float64").to_dict()
    )

    for row, species in enumerate(species_values):
        for col, year in enumerate(years):
            ax = axes[row, col]
            group = summary[
                (summary["species"] == species) & (summary["year"] == year)
            ]
            color = SPECIES_COLORS.get(species, "#666666")
            max_records = float(species_max.get(species, 1.0))
            days_in_year = (
                366
                if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366
                else 365
            )
            month_starts = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
            month_angles = 2.0 * np.pi * (
                month_starts.dayofyear - 1
            ) / days_in_year
            month_labels = [date.strftime("%b") for date in month_starts]
            day_width = 2.0 * np.pi / days_in_year * 0.85

            if not group.empty:
                day_of_year = group["date"].dt.dayofyear.to_numpy()
                angles = 2.0 * np.pi * (day_of_year - 1) / days_in_year
                records = group["presence_count"].to_numpy(dtype="float64")
                ax.bar(
                    angles,
                    records,
                    width=day_width,
                    bottom=0.0,
                    color=color,
                    edgecolor=color,
                    linewidth=0.0,
                    alpha=0.88,
                )

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(month_angles)
            ax.set_xticklabels(month_labels, fontsize=7)
            ax.set_ylim(0, max_records * 1.08)
            radial_ticks = np.linspace(max_records / 4.0, max_records, 4)
            ax.set_yticks(radial_ticks)
            ax.set_yticklabels([f"{tick:.0f}" for tick in radial_ticks], fontsize=7)
            ax.set_rlabel_position(135)
            ax.grid(color="#d6d6d6", linewidth=0.65, alpha=0.8)
            ax.spines["polar"].set_color("#9a9a9a")
            ax.spines["polar"].set_linewidth(0.8)
            if group.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No records",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#666666",
                )
            if row == 0:
                ax.set_title(str(year), fontsize=12, pad=12)
            if col == 0:
                ax.text(
                    -0.18,
                    0.5,
                    species,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=12,
                    fontweight="bold",
                )

    fig.suptitle("Species Data Temporal Coverage", fontsize=15, y=0.995)
    fig.text(0.5, 0.04, "Radial bars show telemetry-derived records per day", ha="center")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.09, wspace=0.34, hspace=0.42)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def main() -> int:
    """Run species temporal coverage plotting."""
    df = load_species_presence()
    summary = daily_summary(df)
    totals = species_totals(summary)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    totals.to_csv(SUMMARY_FILE, index=False)

    saved = plot_temporal_coverage_matrix(summary, OUTPUT_FILE)
    print(f"Saved: {saved}")

    print(f"Saved: {SUMMARY_FILE}")
    print(totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
