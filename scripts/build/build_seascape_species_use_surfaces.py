"""Build observed and predicted species-use summaries by seascape."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path

import duckdb

from riskscape.config import paths
from riskscape.model.dataset import modeling_root


YEARS = "2022"
MODEL_NAME = "som_15x15_hierarchical_k30"
PREDICTION_MODEL = (
    "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30"
)
PREDICTION_PRODUCT = "joint"
SPECIES_YEARS = [2022, 2023]
SURFACE_STAT = "median"

SUMMARY_ROOT = paths["data"] / "plot_exports" / "seascapes"
SURFACE_ROOT = paths["data"] / "modeling" / "seascape_species_use"


def parse_years(years: str) -> list[int]:
    """Parse all, one year, a range, or comma-separated years."""
    if years.lower() == "all":
        return available_years()

    parsed: set[int] = set()
    for part in years.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            parsed.update(range(int(start_text), int(end_text) + 1))
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


def available_years() -> list[int]:
    """Return years with prediction partitions."""
    root = (
        modeling_root("predictions")
        / PREDICTION_MODEL
        / PREDICTION_PRODUCT
    )
    years: list[int] = []

    for path in sorted(root.glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=", maxsplit=1)[1]))

    if not years:
        raise FileNotFoundError(f"No prediction partitions found: {root}")

    return years


def species_training_path(year: int) -> Path:
    """Return species-training partition path."""
    return modeling_root("species_training") / f"year={year}" / "part.parquet"


def prediction_path(year: int) -> Path:
    """Return species-use prediction partition path."""
    return (
        modeling_root("predictions")
        / PREDICTION_MODEL
        / PREDICTION_PRODUCT
        / f"year={year}"
        / "part.parquet"
    )


def seascape_path(year: int, model_name: str) -> Path:
    """Return seascape assignment partition path."""
    return modeling_root("environmental_regimes") / f"year={year}" / "part.parquet"


def surface_path(year: int, model_name: str) -> Path:
    """Return seascape-based species-use surface path."""
    return (
        SURFACE_ROOT
        / model_name
        / f"year={year}"
        / "part.parquet"
    )


def existing(paths_to_check: list[Path]) -> list[str]:
    """Return existing paths as strings."""
    return [str(path) for path in paths_to_check if path.exists()]


def observed_summary_path(model_name: str) -> Path:
    """Return observed-use summary CSV path."""
    return SUMMARY_ROOT / f"observed_species_use_by_seascape_{model_name}.csv"


def predicted_summary_path(model_name: str, years: list[int]) -> Path:
    """Return predicted-use summary CSV path."""
    return (
        SUMMARY_ROOT
        / f"predicted_species_use_by_seascape_{model_name}_{year_label(years)}.csv"
    )


def build_observed_summary(model_name: str) -> Path:
    """Summarize observed positive residence-index rows by species and seascape."""
    species_files = existing([species_training_path(year) for year in SPECIES_YEARS])
    seascape_files = existing([seascape_path(year, model_name) for year in SPECIES_YEARS])

    if not species_files or not seascape_files:
        raise FileNotFoundError("Species-training or seascape partitions are missing")

    query = """
        WITH observed AS (
            SELECT
                CAST(h3 AS UBIGINT) AS h3,
                date,
                species,
                residence_index,
                presence_count,
                individual_count,
                trip_count
            FROM read_parquet($species_files)
            WHERE residence_index > 0
        ),
        joined AS (
            SELECT
                o.species,
                s.seascape,
                o.residence_index,
                o.presence_count,
                o.individual_count,
                o.trip_count,
                s.seascape_distance
            FROM observed AS o
            INNER JOIN read_parquet($seascape_files) AS s
                USING (h3, date)
        ),
        summary AS (
            SELECT
                species,
                CAST(seascape AS INTEGER) AS seascape,
                count(*) AS observed_rows,
                sum(residence_index) AS residence_index_sum,
                avg(residence_index) AS mean_residence_index,
                median(residence_index) AS median_residence_index,
                avg(seascape_distance) AS mean_seascape_distance,
                sum(presence_count) AS presence_count_sum,
                sum(individual_count) AS individual_count_sum,
                sum(trip_count) AS trip_count_sum
            FROM joined
            GROUP BY species, seascape
        )
        SELECT
            *,
            100.0 * observed_rows
                / sum(observed_rows) OVER (PARTITION BY species)
                AS observed_row_percent
        FROM summary
        ORDER BY species, seascape
    """

    out_file = observed_summary_path(model_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(database=":memory:") as con:
        df = con.execute(
            query,
            {
                "species_files": species_files,
                "seascape_files": seascape_files,
            },
        ).fetchdf()

    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    return out_file


def build_predicted_summary(model_name: str, years: list[int]) -> Path:
    """Summarize predicted species use by species and seascape."""
    prediction_files = existing([prediction_path(year) for year in years])
    seascape_files = existing([seascape_path(year, model_name) for year in years])

    if not prediction_files or not seascape_files:
        raise FileNotFoundError("Prediction or seascape partitions are missing")

    query = """
        SELECT
            p.species,
            CAST(s.seascape AS INTEGER) AS seascape,
            count(*) AS cell_days,
            avg(p.species_use_log_pred) AS mean_log_residence_index,
            median(p.species_use_log_pred) AS median_log_residence_index,
            median(
                CASE
                    WHEN p.species_use_log_pred > 0
                    THEN p.species_use_log_pred
                    ELSE NULL
                END
            ) AS non_zero_median_log_residence_index,
            avg(exp(p.species_use_log_pred) - 1.0) AS mean_residence_index,
            median(exp(p.species_use_log_pred) - 1.0) AS median_residence_index,
            median(
                CASE
                    WHEN p.species_use_log_pred > 0
                    THEN exp(p.species_use_log_pred) - 1.0
                    ELSE NULL
                END
            ) AS non_zero_median_residence_index,
            avg(CASE WHEN p.species_use_log_pred > 0 THEN 1.0 ELSE 0.0 END)
                AS predicted_positive_fraction,
            avg(s.seascape_distance) AS mean_seascape_distance
        FROM read_parquet($prediction_files) AS p
        INNER JOIN read_parquet($seascape_files) AS s
            USING (h3, date)
        GROUP BY p.species, s.seascape
        ORDER BY p.species, s.seascape
    """

    out_file = predicted_summary_path(model_name, years)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(database=":memory:") as con:
        df = con.execute(
            query,
            {
                "prediction_files": prediction_files,
                "seascape_files": seascape_files,
            },
        ).fetchdf()

    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    return out_file


def build_surface_year(
    year: int,
    model_name: str,
    observed_file: Path,
    predicted_file: Path,
    surface_stat: str,
) -> Path:
    """Write a yearly seascape-based species-use surface."""
    if surface_stat not in {"median", "mean"}:
        raise ValueError("--surface-stat must be median or mean")

    log_col = f"{surface_stat}_log_residence_index"
    raw_col = f"{surface_stat}_residence_index"
    prediction_file = prediction_path(year)
    seascape_file = seascape_path(year, model_name)

    if not prediction_file.exists() or not seascape_file.exists():
        raise FileNotFoundError(f"Missing prediction or seascape partition for {year}")

    out_file = surface_path(year, model_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    query = f"""
        COPY (
            SELECT
                p.h3,
                CAST(p.date AS TIMESTAMP) AS date,
                p.species,
                CAST(s.seascape AS INTEGER) AS seascape,
                s.seascape_distance,
                obs.observed_rows,
                obs.observed_row_percent,
                obs.mean_residence_index AS observed_mean_residence_index,
                obs.median_residence_index AS observed_median_residence_index,
                pred.cell_days AS predicted_summary_cell_days,
                pred.{log_col} AS seascape_species_use_log_pred,
                pred.{raw_col} AS seascape_species_use_residence_index,
                pred.mean_log_residence_index AS seascape_mean_log_residence_index,
                pred.median_log_residence_index AS seascape_median_log_residence_index,
                pred.non_zero_median_log_residence_index
                    AS seascape_non_zero_median_log_residence_index,
                pred.mean_residence_index AS seascape_mean_residence_index,
                pred.median_residence_index AS seascape_median_residence_index,
                pred.non_zero_median_residence_index
                    AS seascape_non_zero_median_residence_index,
                pred.predicted_positive_fraction
            FROM read_parquet($prediction_file) AS p
            INNER JOIN read_parquet($seascape_file) AS s
                USING (h3, date)
            LEFT JOIN read_csv_auto($observed_file) AS obs
                ON p.species = obs.species
                AND s.seascape = obs.seascape
            LEFT JOIN read_csv_auto($predicted_file) AS pred
                ON p.species = pred.species
                AND s.seascape = pred.seascape
        )
        TO '{out_file.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    with duckdb.connect(database=":memory:") as con:
        con.execute(
            query,
            {
                "prediction_file": str(prediction_file),
                "seascape_file": str(seascape_file),
                "observed_file": str(observed_file),
                "predicted_file": str(predicted_file),
            },
        )

    print(f"Saved: {out_file}")

    return out_file


def build_surfaces(
    years: list[int],
    model_name: str,
    observed_file: Path,
    predicted_file: Path,
    surface_stat: str,
) -> list[Path]:
    """Build seascape species-use surfaces for selected years."""
    return [
        build_surface_year(
            year=year,
            model_name=model_name,
            observed_file=observed_file,
            predicted_file=predicted_file,
            surface_stat=surface_stat,
        )
        for year in years
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize observed and predicted species use by seascape and "
            "write seascape-based species-use surfaces for risk calculations."
        )
    )
    parser.add_argument(
        "--years",
        default=YEARS,
        help="Prediction years: all, one year, a range, or comma-separated years.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--surface-stat",
        default=SURFACE_STAT,
        choices=("median", "mean"),
        help=(
            "Statistic used as the canonical seascape species-use estimate. "
            "Non-zero summaries are retained as supporting columns only."
        ),
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Only write observed and predicted seascape summaries.",
    )
    parser.add_argument(
        "--surfaces-only",
        action="store_true",
        help="Only write surfaces using existing summary CSV files.",
    )

    return parser.parse_args()


def main() -> int:
    """Run seascape species-use summary and surface workflow."""
    args = parse_args()
    years = parse_years(args.years)

    observed_file = observed_summary_path(args.model_name)
    predicted_file = predicted_summary_path(args.model_name, years)

    if not args.surfaces_only:
        observed_file = build_observed_summary(args.model_name)
        predicted_file = build_predicted_summary(args.model_name, years)

    if not args.summaries_only:
        build_surfaces(
            years=years,
            model_name=args.model_name,
            observed_file=observed_file,
            predicted_file=predicted_file,
            surface_stat=args.surface_stat,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
