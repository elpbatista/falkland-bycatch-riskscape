"""Quantify seascape class balance and species-training support.

This script compares candidate environmental seascape designs as validation
blocks. It is intentionally separate from model fitting: the outputs help decide
which candidates are worth using in BlockCV model runs.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import SPECIES_TARGET


METRICS_ROOT = paths["data"] / "modeling" / "metrics" / "seascapes"
SPECIES_TABLE = paths["data"] / "modeling" / "species_training"


@dataclass(frozen=True)
class Candidate:
    """One seascape validation design to summarize."""

    name: str
    table: str
    class_column: str


CANDIDATES = [
    Candidate(
        name="som_k30",
        table="environmental_regimes",
        class_column="seascape",
    ),
]


def quote_identifier(name: str) -> str:
    """Return a DuckDB-safe identifier."""
    return '"' + name.replace('"', '""') + '"'


def table_root(table: str) -> Path:
    """Return a modeling table root."""
    return paths["data"] / "modeling" / table


def available_years(root: Path) -> list[int]:
    """Return years with yearly parquet partitions."""
    years: list[int] = []
    for path in sorted(root.glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=", maxsplit=1)[1]))
    return years


def year_files(root: Path, years: list[int]) -> list[str]:
    """Return existing yearly parquet paths."""
    return [
        str(root / f"year={year}" / "part.parquet")
        for year in years
        if (root / f"year={year}" / "part.parquet").exists()
    ]


def normalized_entropy(counts: pd.Series) -> float:
    """Return entropy scaled to [0, 1]."""
    values = counts.to_numpy(dtype="float64")
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(values)))


def assignment_monthly_counts(candidate: Candidate) -> pd.DataFrame:
    """Return assignment counts by candidate/year/month/class."""
    root = table_root(candidate.table)
    years = available_years(root)
    files = year_files(root, years)
    if not files:
        raise FileNotFoundError(f"No partitions found for {candidate.table}")

    class_expr = quote_identifier(candidate.class_column)
    query = f"""
        SELECT
            CAST(year AS INTEGER) AS year,
            CAST(month(date) AS INTEGER) AS month,
            CAST({class_expr} AS INTEGER) AS seascape,
            count(*) AS n_cell_days
        FROM read_parquet($files)
        GROUP BY year, month, seascape
        ORDER BY year, month, seascape
    """
    with duckdb.connect(database=":memory:") as con:
        out = con.execute(query, {"files": files}).fetchdf()

    out.insert(0, "candidate", candidate.name)
    out.insert(1, "table", candidate.table)
    return out


def summarize_assignment_balance(monthly: pd.DataFrame) -> dict[str, object]:
    """Summarize assignment balance over all available H3/date rows."""
    class_counts = monthly.groupby("seascape", observed=True)["n_cell_days"].sum()
    monthly_totals = monthly.groupby(["year", "month"], observed=True)["n_cell_days"].sum()
    monthly_max = (
        monthly.groupby(["year", "month"], observed=True)["n_cell_days"].max()
        / monthly_totals
    )
    classes_per_month = monthly.groupby(["year", "month"], observed=True)[
        "seascape"
    ].nunique()
    years = sorted(monthly["year"].unique().tolist())

    return {
        "years": f"{min(years)}-{max(years)}" if years else "",
        "n_years": len(years),
        "n_year_months": int(monthly[["year", "month"]].drop_duplicates().shape[0]),
        "n_classes": int(class_counts.shape[0]),
        "total_cell_days": int(class_counts.sum()),
        "min_class_cell_days": int(class_counts.min()),
        "median_class_cell_days": float(class_counts.median()),
        "max_class_cell_days": int(class_counts.max()),
        "class_balance_ratio": float(class_counts.min() / class_counts.max()),
        "class_entropy": normalized_entropy(class_counts),
        "min_classes_per_month": int(classes_per_month.min()),
        "median_classes_per_month": float(classes_per_month.median()),
        "mean_dominant_monthly_share": float(monthly_max.mean()),
        "max_dominant_monthly_share": float(monthly_max.max()),
    }


def species_support(candidate: Candidate) -> pd.DataFrame:
    """Return species-training row and positive support by seascape class."""
    candidate_root = table_root(candidate.table)
    species_years = available_years(SPECIES_TABLE)
    candidate_years = available_years(candidate_root)
    years = sorted(set(species_years) & set(candidate_years))
    if not years:
        raise FileNotFoundError(
            f"No overlap between species_training and {candidate.table}"
        )

    species_files = year_files(SPECIES_TABLE, years)
    candidate_files = year_files(candidate_root, years)
    class_expr = quote_identifier(candidate.class_column)
    query = f"""
        WITH blocks AS (
            SELECT
                h3,
                CAST(date AS DATE) AS date,
                CAST({class_expr} AS INTEGER) AS seascape
            FROM read_parquet($candidate_files)
        )
        SELECT
            s.species,
            b.seascape,
            count(*) AS n_rows,
            sum(CASE WHEN s.{SPECIES_TARGET} > 0 THEN 1 ELSE 0 END) AS n_positive_rows,
            avg(s.{SPECIES_TARGET}) AS mean_species_use
        FROM read_parquet($species_files) AS s
        INNER JOIN blocks AS b
            ON s.h3 = b.h3
            AND CAST(s.date AS DATE) = b.date
        GROUP BY s.species, b.seascape
        ORDER BY s.species, b.seascape
    """
    with duckdb.connect(database=":memory:") as con:
        out = con.execute(
            query,
            {
                "species_files": species_files,
                "candidate_files": candidate_files,
            },
        ).fetchdf()

    out.insert(0, "candidate", candidate.name)
    out.insert(1, "table", candidate.table)
    out.insert(2, "years", ",".join(str(year) for year in years))
    return out


def summarize_species_support(
    support: pd.DataFrame,
    n_classes: int,
) -> pd.DataFrame:
    """Summarize species support by candidate/species."""
    rows: list[dict[str, object]] = []
    for species, group in support.groupby("species", observed=True):
        positives = group["n_positive_rows"] > 0
        rows.append(
            {
                "species": species,
                "species_training_rows": int(group["n_rows"].sum()),
                "species_positive_rows": int(group["n_positive_rows"].sum()),
                "classes_with_rows": int(group["seascape"].nunique()),
                "classes_without_rows": int(n_classes - group["seascape"].nunique()),
                "classes_with_positives": int(positives.sum()),
                "classes_without_positives": int(n_classes - positives.sum()),
                "min_rows_per_observed_class": int(group["n_rows"].min()),
                "median_rows_per_observed_class": float(group["n_rows"].median()),
                "min_positive_rows_per_positive_class": int(
                    group.loc[positives, "n_positive_rows"].min()
                )
                if positives.any()
                else 0,
                "median_positive_rows_per_positive_class": float(
                    group.loc[positives, "n_positive_rows"].median()
                )
                if positives.any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """Run seascape validation-design diagnostics."""
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    balance_rows: list[dict[str, object]] = []
    monthly_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []
    support_summary_frames: list[pd.DataFrame] = []

    for candidate in CANDIDATES:
        monthly = assignment_monthly_counts(candidate)
        monthly_frames.append(monthly)
        balance = summarize_assignment_balance(monthly)
        balance["candidate"] = candidate.name
        balance["table"] = candidate.table
        balance_rows.append(balance)

        support = species_support(candidate)
        support_frames.append(support)
        support_summary = summarize_species_support(
            support,
            n_classes=int(balance["n_classes"]),
        )
        support_summary.insert(0, "candidate", candidate.name)
        support_summary.insert(1, "table", candidate.table)
        support_summary_frames.append(support_summary)

    balance_df = pd.DataFrame(balance_rows)
    monthly_df = pd.concat(monthly_frames, ignore_index=True)
    support_df = pd.concat(support_frames, ignore_index=True)
    support_summary_df = pd.concat(support_summary_frames, ignore_index=True)

    balance_file = METRICS_ROOT / "seascape_validation_design_balance.csv"
    monthly_file = METRICS_ROOT / "seascape_validation_design_monthly_counts.csv"
    support_file = METRICS_ROOT / "seascape_validation_design_species_support.csv"
    support_summary_file = (
        METRICS_ROOT / "seascape_validation_design_species_support_summary.csv"
    )

    balance_df.to_csv(balance_file, index=False)
    monthly_df.to_csv(monthly_file, index=False)
    support_df.to_csv(support_file, index=False)
    support_summary_df.to_csv(support_summary_file, index=False)

    print(f"Saved balance: {balance_file}")
    print(f"Saved monthly counts: {monthly_file}")
    print(f"Saved species support: {support_file}")
    print(f"Saved species support summary: {support_summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
