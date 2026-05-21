"""Plot weekly gear-aware realized-risk examples.

This script keeps the core prediction and fishing-effort feature tables
unchanged. It joins the 2022 hybrid species-use predictions to the separate
H3/date/gear/flag fishing-activity export and plots selected ISO weeks for
species/gear pairs used as operational examples.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_prediction_maps import (  # noqa: E402
    AGG,
    prediction_path,
)
from plot_weekly_operator_latent_risk import REPRESENTATIVE_WEEKS  # noqa: E402
from plot_weekly_operator_fisheries_grid_example import (  # noqa: E402
    build_h3_to_fisheries_lookup,
    load_reference_overlays,
    plot_small_multiples,
)
from riskscape.config import paths  # noqa: E402
from riskscape.grid import load_grid  # noqa: E402
from riskscape.visualization.maps import aggregation_name  # noqa: E402


YEAR = 2022
MODEL_NAME = (
    "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_"
    "bayesian_gmm_k30"
)
PRODUCT_NAME = "joint"
PAIRS = (
    ("BBAL", "SET_LONGLINES"),
    ("SAFS", "TRAWLERS"),
)
OUTPUT_ROOT = paths["plots"] / "fishing_activity"
SUMMARY_ROOT = paths["data"] / "plot_exports" / "fishing_activity"
VALUE_COL = "display_latent_risk_log_pred_mean"


def exposure_path(year: int) -> Path:
    """Return the gear/flag exposure export for one year."""
    return (
        paths["data"]
        / "plot_exports"
        / "fishing_activity"
        / f"fishing_effort_by_gear_flag_{year}.parquet"
    )


def load_weekly_gear_risk(
    year: int,
    model_name: str,
    product_name: str,
    pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Join predictions to gear-filtered effort and aggregate by ISO week."""
    pred_path = prediction_path(year, model_name, product_name)
    effort_path = exposure_path(year)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction partition not found: {pred_path}")
    if not effort_path.exists():
        raise FileNotFoundError(f"Gear/flag exposure table not found: {effort_path}")

    pair_rows = ", ".join(
        f"('{species}', '{gear_type}')" for species, gear_type in pairs
    )
    query = f"""
        WITH selected_pairs(species, gear_type) AS (
            VALUES {pair_rows}
        ),
        exposure AS (
            SELECT
                CAST(h3 AS UBIGINT) AS h3,
                CAST(date AS DATE) AS date,
                gear_type,
                SUM(fishing_hours)::DOUBLE AS fishing_hours,
                SUM(vessel_count)::DOUBLE AS vessel_count
            FROM read_parquet($effort_path)
            WHERE gear_type IN (SELECT DISTINCT gear_type FROM selected_pairs)
            GROUP BY h3, date, gear_type
        ),
        daily_risk AS (
            SELECT
                CAST(p.h3 AS UBIGINT) AS h3,
                CAST(p.date AS DATE) AS date,
                CAST(date_part('isoyear', CAST(p.date AS DATE)) AS INTEGER)
                    AS iso_year,
                CAST(date_part('week', CAST(p.date AS DATE)) AS INTEGER)
                    AS iso_week,
                p.species,
                e.gear_type,
                p.species_use_log_pred,
                CAST(e.fishing_hours * e.vessel_count AS FLOAT)
                    AS fishing_activity,
                CAST(
                    p.species_use_log_pred
                    + ln(1.0 + e.fishing_hours * e.vessel_count)
                    AS FLOAT
                ) AS risk_log_pred
            FROM read_parquet($pred_path) p
            INNER JOIN selected_pairs s
            ON p.species = s.species
            INNER JOIN exposure e
            ON CAST(p.h3 AS UBIGINT) = e.h3
            AND CAST(p.date AS DATE) = e.date
            AND s.gear_type = e.gear_type
        )
        SELECT
            h3,
            iso_year,
            iso_week,
            species,
            gear_type,
            COUNT(*)::INTEGER AS n_days,
            AVG(species_use_log_pred)::FLOAT AS species_use_log_pred,
            SUM(fishing_activity)::FLOAT AS fishing_activity,
            COALESCE(
                AVG(CASE WHEN risk_log_pred > 0 THEN risk_log_pred ELSE NULL END),
                0.0
            )::FLOAT AS risk_log_pred
        FROM daily_risk
        WHERE iso_year = $year
        GROUP BY h3, iso_year, iso_week, species, gear_type
        ORDER BY species, gear_type, iso_week, h3
    """
    with duckdb.connect(database=":memory:") as con:
        out = con.execute(
            query,
            {
                "pred_path": str(pred_path),
                "effort_path": str(effort_path),
                "year": year,
            },
        ).fetchdf()

    out["h3"] = out["h3"].astype("uint64")
    return out


def summarize_weeks(weekly_risk: pd.DataFrame) -> pd.DataFrame:
    """Summarize weekly risk values for representative-week selection."""
    positive = weekly_risk[weekly_risk["risk_log_pred"] > 0].copy()
    if positive.empty:
        raise ValueError("No positive weekly gear-aware risk values found")

    summary = (
        positive.groupby(["species", "gear_type", "iso_year", "iso_week"])
        .agg(
            n_h3=("h3", "nunique"),
            mean_risk=("risk_log_pred", "mean"),
            q90_risk=("risk_log_pred", lambda values: values.quantile(0.90)),
            max_risk=("risk_log_pred", "max"),
            total_fishing_activity=("fishing_activity", "sum"),
        )
        .reset_index()
        .sort_values(["species", "gear_type", "q90_risk"], ascending=[True, True, False])
    )
    return summary


def aggregate_to_fisheries_grid(
    weekly_risk: pd.DataFrame,
    h3_to_fisheries: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate weekly gear-aware H3 risk to fisheries grid squares."""
    joined = weekly_risk.merge(h3_to_fisheries, on="h3", how="inner")
    out = (
        joined.groupby(
            ["fisheries_grid", "species", "gear_type", "iso_week"],
            as_index=False,
        )
        .agg(
            n_h3=("h3", "nunique"),
            risk_log_pred_mean=("risk_log_pred", "mean"),
        )
        .sort_values(["species", "gear_type", "iso_week", "fisheries_grid"])
    )
    out["latent_risk_log_pred_mean"] = out["risk_log_pred_mean"]
    out[VALUE_COL] = out["risk_log_pred_mean"].where(out["risk_log_pred_mean"] > 0)
    out["source_species"] = out["species"]
    out["species"] = out["source_species"] + " - " + out["gear_type"]
    return out


def selected_existing_weeks(
    summary: pd.DataFrame,
    weeks: tuple[int, ...],
) -> pd.DataFrame:
    """Return the existing weekly-operator example weeks for each pair."""
    selected_frames = []
    week_values = set(weeks)
    for (species, gear_type), group in summary.groupby(["species", "gear_type"]):
        selected = group[group["iso_week"].isin(week_values)].copy()
        if selected.empty:
            raise ValueError(f"No existing example weeks found for {species} {gear_type}")
        selected_frames.append(selected)

    out = pd.concat(selected_frames, ignore_index=True)
    out = out.sort_values(["species", "gear_type", "iso_week"]).reset_index(drop=True)
    return out


def write_summaries(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    year: int,
) -> tuple[Path, Path]:
    """Write weekly summaries used to justify the selected weeks."""
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_ROOT / f"gear_aware_weekly_risk_summary_{year}.csv"
    selected_path = SUMMARY_ROOT / f"gear_aware_representative_iso_weeks_{year}.csv"
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    return summary_path, selected_path


def output_file(
    year: int,
) -> Path:
    """Return the small-multiples output path."""
    agg_name = aggregation_name(AGG)
    return (
        OUTPUT_ROOT
        / (
            "gear_aware_weekly_realized_risk_fisheries_grid_example_"
            f"{agg_name}_{year}.png"
        )
    )


def write_fisheries_aggregate(aggregated: pd.DataFrame, year: int) -> Path:
    """Write the fisheries-grid aggregate behind the example map."""
    out_file = (
        SUMMARY_ROOT
        / (
            "gear_aware_weekly_realized_risk_fisheries_grid_"
            f"{year}.parquet"
        )
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_parquet(out_file, index=False)
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot weekly gear-aware realized-risk example maps."
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=list(REPRESENTATIVE_WEEKS),
        help="ISO weeks to plot; defaults to the existing weekly-operator example weeks.",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run weekly gear-aware risk summaries and maps."""
    args = parse_args()
    weekly_risk = load_weekly_gear_risk(
        year=args.year,
        model_name=MODEL_NAME,
        product_name=PRODUCT_NAME,
        pairs=PAIRS,
    )
    summary = summarize_weeks(weekly_risk)
    selected = selected_existing_weeks(
        summary=summary,
        weeks=tuple(args.weeks),
    )
    summary_path, selected_path = write_summaries(summary, selected, args.year)
    print("Saved weekly summary:", summary_path)
    print("Saved representative weeks:", selected_path)
    print(selected[["species", "gear_type", "iso_week", "n_h3", "q90_risk", "max_risk"]])

    if args.summary_only:
        return 0

    h3_grid = load_grid(uint64=True)
    fisheries_grid, limits = load_reference_overlays()
    h3_to_fisheries = build_h3_to_fisheries_lookup(h3_grid, fisheries_grid)
    aggregated = aggregate_to_fisheries_grid(weekly_risk, h3_to_fisheries)
    aggregate_path = write_fisheries_aggregate(aggregated, args.year)
    print("Saved fisheries-grid aggregate:", aggregate_path)

    species_gear_values = [
        f"{species} - {gear_type}" for species, gear_type in PAIRS
    ]
    saved = plot_small_multiples(
        aggregated=aggregated,
        fisheries_grid=fisheries_grid,
        limits=limits,
        species_values=species_gear_values,
        weeks=list(args.weeks),
        model_name=MODEL_NAME,
        product_name=PRODUCT_NAME,
        start_year=args.year,
        end_year=args.year,
        title=f"Weekly Gear-Filtered Realized Risk on Fisheries Grid - {args.year}",
        out_file=output_file(args.year),
        value_col=VALUE_COL,
        style_species_values=sorted({species for species, _ in PAIRS}),
    )
    print("Saved map:", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
