"""Build weekly latent-risk products for operator-facing maps.

The script writes two H3/species weekly data products:

- ISO-week climatology averaged across the selected year range.
- A single-year ISO-week sequence for animation or small multiples.

The product is latent risk, not realized risk:

    latent_risk_log_pred = species_use_log_pred + log1p(minimum_effort_unit)

Rows also carry ``display_latent_risk_log_pred``. That field is null where the
weekly species-use mean is below the shared display threshold used by the
prediction maps.
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

import duckdb
import numpy as np

from riskscape.config import paths
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_MIN_DISPLAY,
)


MODEL_NAME = "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30"
PRODUCT_NAME = "joint"
SPECIES = ("BBAL", "SAFS")
START_YEAR = 2014
END_YEAR = 2023
SEQUENCE_YEAR = 2022
OUTPUT_ROOT = paths["data"] / "modeling" / "weekly_operator"
SUMMARY_ROOT = paths["data"] / "plot_exports" / "weekly_operator"


def prediction_glob(model_name: str, product_name: str) -> Path:
    """Return a glob path for all yearly prediction partitions."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / model_name
        / product_name
        / "year=*"
        / "part.parquet"
    )


def year_label(start_year: int, end_year: int) -> str:
    """Return compact year-range label."""
    return f"{start_year}-{end_year}"


def output_paths(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
    sequence_year: int,
) -> dict[str, Path]:
    """Return output paths for the weekly products."""
    root = OUTPUT_ROOT / model_name / product_name
    label = year_label(start_year, end_year)
    return {
        "climatology": root
        / f"latent_risk_iso_week_climatology_{label}.parquet",
        "sequence": root
        / f"latent_risk_iso_week_sequence_{sequence_year}.parquet",
        "weekly_by_year": root
        / f"latent_risk_iso_week_by_year_{label}.parquet",
    }


def summary_paths(
    model_name: str,
    product_name: str,
    start_year: int,
    end_year: int,
    sequence_year: int,
) -> dict[str, Path]:
    """Return compact CSV diagnostic paths."""
    root = SUMMARY_ROOT / model_name / product_name
    label = year_label(start_year, end_year)
    return {
        "climatology": root
        / f"latent_risk_iso_week_climatology_summary_{label}.csv",
        "sequence": root
        / f"latent_risk_iso_week_sequence_summary_{sequence_year}.csv",
    }


def species_sql(species: tuple[str, ...]) -> str:
    """Return SQL literal list for species codes."""
    return ", ".join(f"'{item}'" for item in species)


def sql_literal(value: str | Path) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def remove_existing(path_values: list[Path], overwrite: bool) -> None:
    """Remove existing output files when overwriting is requested."""
    for path in path_values:
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output exists: {path}. Use --overwrite to replace it."
                )
            path.unlink()


def build_products(args: argparse.Namespace) -> None:
    """Build weekly latent-risk parquet and summary CSV products."""
    input_path = prediction_glob(args.model_name, args.product_name)
    if not list(input_path.parent.parent.glob("year=*/part.parquet")):
        raise FileNotFoundError(f"No prediction partitions found: {input_path}")

    outputs = output_paths(
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
        sequence_year=args.sequence_year,
    )
    summaries = summary_paths(
        model_name=args.model_name,
        product_name=args.product_name,
        start_year=args.start_year,
        end_year=args.end_year,
        sequence_year=args.sequence_year,
    )

    for path in [*outputs.values(), *summaries.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)

    remove_existing([*outputs.values(), *summaries.values()], args.overwrite)

    baseline = float(np.log1p(args.minimum_effort_unit))
    species_filter = species_sql(tuple(args.species))
    input_literal = sql_literal(input_path)

    weekly_query = f"""
        SELECT
            CAST(h3 AS UBIGINT) AS h3,
            species,
            CAST(date_part('isoyear', CAST(date AS DATE)) AS INTEGER) AS iso_year,
            CAST(date_part('week', CAST(date AS DATE)) AS INTEGER) AS iso_week,
            COUNT(*)::INTEGER AS n_days,
            AVG(species_use_log_pred)::FLOAT AS species_use_log_pred_mean,
            (AVG(species_use_log_pred) + {baseline})::FLOAT
                AS latent_risk_log_pred_mean,
            CASE
                WHEN AVG(species_use_log_pred) > {args.species_use_log_min_display}
                THEN (AVG(species_use_log_pred) + {baseline})::FLOAT
                ELSE NULL
            END AS display_latent_risk_log_pred_mean
        FROM read_parquet({input_literal}, hive_partitioning=false)
        WHERE species IN ({species_filter})
        GROUP BY h3, species, iso_year, iso_week
        HAVING iso_year BETWEEN {args.start_year} AND {args.end_year}
    """

    sequence_query = f"""
        SELECT *
        FROM weekly_by_year
        WHERE iso_year = {args.sequence_year}
        ORDER BY species, iso_week, h3
    """

    climatology_query = f"""
        SELECT
            h3,
            species,
            iso_week,
            COUNT(DISTINCT iso_year)::INTEGER AS n_years,
            SUM(n_days)::INTEGER AS n_days,
            AVG(species_use_log_pred_mean)::FLOAT
                AS species_use_log_pred_mean,
            AVG(latent_risk_log_pred_mean)::FLOAT
                AS latent_risk_log_pred_mean,
            CASE
                WHEN AVG(species_use_log_pred_mean)
                    > {args.species_use_log_min_display}
                THEN AVG(latent_risk_log_pred_mean)::FLOAT
                ELSE NULL
            END AS display_latent_risk_log_pred_mean
        FROM weekly_by_year
        GROUP BY h3, species, iso_week
        ORDER BY species, iso_week, h3
    """

    climatology_summary_query = """
        SELECT
            species,
            iso_week,
            COUNT(*)::INTEGER AS n_h3,
            COUNT(display_latent_risk_log_pred_mean)::INTEGER AS n_display_h3,
            AVG(latent_risk_log_pred_mean)::FLOAT AS mean_latent_risk,
            MAX(latent_risk_log_pred_mean)::FLOAT AS max_latent_risk
        FROM climatology
        GROUP BY species, iso_week
        ORDER BY species, iso_week
    """

    sequence_summary_query = """
        SELECT
            species,
            iso_week,
            COUNT(*)::INTEGER AS n_h3,
            COUNT(display_latent_risk_log_pred_mean)::INTEGER AS n_display_h3,
            AVG(latent_risk_log_pred_mean)::FLOAT AS mean_latent_risk,
            MAX(latent_risk_log_pred_mean)::FLOAT AS max_latent_risk
        FROM sequence
        GROUP BY species, iso_week
        ORDER BY species, iso_week
    """

    with duckdb.connect(database=":memory:") as con:
        con.execute("CREATE TEMP VIEW weekly_by_year AS " + weekly_query)
        con.execute("CREATE TEMP VIEW sequence AS " + sequence_query)
        con.execute("CREATE TEMP VIEW climatology AS " + climatology_query)

        con.execute(
            f"COPY weekly_by_year TO '{outputs['weekly_by_year']}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY sequence TO '{outputs['sequence']}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY climatology TO '{outputs['climatology']}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY ({climatology_summary_query}) TO "
            f"'{summaries['climatology']}' (HEADER, DELIMITER ',')"
        )
        con.execute(
            f"COPY ({sequence_summary_query}) TO "
            f"'{summaries['sequence']}' (HEADER, DELIMITER ',')"
        )

    for label, path in outputs.items():
        print(f"Saved {label}: {path}")
    for label, path in summaries.items():
        print(f"Saved {label} summary: {path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build weekly latent-risk operator products.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--product-name", default=PRODUCT_NAME)
    parser.add_argument("--species", nargs="+", default=list(SPECIES))
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--sequence-year", type=int, default=SEQUENCE_YEAR)
    parser.add_argument(
        "--minimum-effort-unit",
        type=float,
        default=MINIMUM_EFFORT_UNIT,
    )
    parser.add_argument(
        "--species-use-log-min-display",
        type=float,
        default=SPECIES_USE_LOG_MIN_DISPLAY,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the weekly operator product build."""
    build_products(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
