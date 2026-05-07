"""Build the combined Big Beautiful Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from riskscape.config import paths


YEAR_START = 2014
YEAR_END = 2023
PREDICTION_MODEL = "hybrid_presence_gate_extra_trees_bayesian_gmm"
PLAUSIBILITY_MODEL = "bayesian_gmm"
PRODUCT_NAME = "joint"
OUTPUT_TABLE = "big_beautiful_dataset"

KEYS = ("h3", "date", "species")
GRID_KEYS = ("h3", "date")
SPECIES_OBSERVATION_COLS = (
    "residence_index",
    "presence_count",
    "individual_count",
    "trip_count",
)


def grid_path() -> Path:
    """Return the uint64 H3 grid path."""
    return paths["grids"] / "h3_res6_falkland_islands_uint64.parquet"


def seasonal_lookup_path() -> Path:
    """Return the processed seasonal lookup path."""
    return paths["data"] / "processed" / "seasonal_lookup.parquet"


def modeling_path(table: str, year: int) -> Path:
    """Return one modeling partition path."""
    return paths["data"] / "modeling" / table / f"year={year}" / "part.parquet"


def prediction_path(year: int) -> Path:
    """Return one prediction partition path."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / PREDICTION_MODEL
        / PRODUCT_NAME
        / f"year={year}"
        / "part.parquet"
    )


def plausibility_path(year: int) -> Path:
    """Return one plausibility partition path."""
    return (
        paths["data"]
        / "modeling"
        / "plausibility"
        / PLAUSIBILITY_MODEL
        / PRODUCT_NAME
        / f"year={year}"
        / "part.parquet"
    )


def output_path(year: int) -> Path:
    """Return one combined dataset output path."""
    return modeling_path(OUTPUT_TABLE, year)


def require_path(path: Path, label: str) -> None:
    """Raise if a required input is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def sql_path(path: Path) -> str:
    """Return a SQL-safe path literal."""
    return str(path).replace("'", "''")


def parquet_sql(path: Path) -> str:
    """Return a DuckDB parquet read expression without hive partition columns."""
    return f"read_parquet('{sql_path(path)}', hive_partitioning=false)"


def parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Return parquet column names without loading the table."""
    require_path(path, "parquet")
    rows = con.execute(
        f"DESCRIBE SELECT * FROM {parquet_sql(path)}"
    ).fetchall()
    return [row[0] for row in rows]


def qualified_columns(
    columns: list[str],
    alias: str,
    exclude: set[str],
) -> list[str]:
    """Return qualified select expressions excluding repeated columns."""
    return [f"{alias}.{col}" for col in columns if col not in exclude]


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Return parquet row count."""
    require_path(path, "parquet")
    return int(
        con.execute(
            f"SELECT count(*) FROM {parquet_sql(path)}"
        ).fetchone()[0]
    )


def missing_required_inputs(year: int) -> list[Path]:
    """Return required input paths that are missing for a year."""
    required = [
        prediction_path(year),
        plausibility_path(year),
        modeling_path("feature_grid", year),
        modeling_path("fishing_training", year),
        grid_path(),
        seasonal_lookup_path(),
    ]
    return [path for path in required if not path.exists()]


def species_select(species_path: Path) -> list[str]:
    """Return species observation select expressions."""
    if not species_path.exists():
        return [
            "CAST(NULL AS FLOAT) AS residence_index",
            "CAST(NULL AS INTEGER) AS presence_count",
            "CAST(NULL AS INTEGER) AS individual_count",
            "CAST(NULL AS INTEGER) AS trip_count",
            "FALSE AS species_observed",
        ]

    return [
        "s.residence_index",
        "s.presence_count",
        "s.individual_count",
        "s.trip_count",
        "s.residence_index IS NOT NULL AS species_observed",
    ]


def species_join(species_path: Path) -> str:
    """Return the optional species-training join SQL."""
    if not species_path.exists():
        return ""

    return (
        f"LEFT JOIN {parquet_sql(species_path)} AS s "
        "USING (h3, date, species)"
    )


def adjusted_doy_sql(date_expr: str) -> str:
    """Return SQL expression for leap-year-adjusted day of year."""
    year_expr = f"year({date_expr})"
    doy_expr = f"dayofyear({date_expr})"
    is_leap = (
        f"({year_expr} % 4 = 0 AND "
        f"({year_expr} % 100 != 0 OR {year_expr} % 400 = 0))"
    )
    return (
        f"CAST({doy_expr} - CASE WHEN {is_leap} AND {doy_expr} > 59 "
        "THEN 1 ELSE 0 END AS SMALLINT)"
    )


def prediction_select(pred_cols: list[str]) -> list[str]:
    """Return prediction columns in their native model output scale."""
    select_cols = [
        "p.h3",
        "p.date",
        "p.species",
        "p.species_use_log_pred",
        "p.hybrid_alpha",
        "p.species_use_ml_log_pred",
        "p.fishing_activity_log",
        "p.risk_log_pred",
    ]

    if "plausibility" in pred_cols:
        select_cols.append("p.plausibility")
    else:
        select_cols.append("pl.plausibility")

    return select_cols


def feature_select(feature_cols: list[str]) -> list[str]:
    """Return interpretable environmental/static feature expressions."""
    excluded = set(GRID_KEYS) | {
        "chl_log",
        "doy_sin",
        "doy_cos",
        "lat_sin",
        "lat_cos",
        "lon_sin",
        "lon_cos",
    }
    select_cols = qualified_columns(feature_cols, "fg", exclude=excluded)

    if "chl_log" in feature_cols:
        select_cols.insert(3, "CAST(exp(fg.chl_log) - 1 AS REAL) AS chl")

    select_cols.append("sl.adjusted_doy")
    select_cols.append("g.lat")
    select_cols.append("g.lon")

    return select_cols


def build_query(con: duckdb.DuckDBPyConnection, year: int) -> str:
    """Return SQL for one combined dataset partition."""
    pred_path = prediction_path(year)
    plaus_path = plausibility_path(year)
    feature_path = modeling_path("feature_grid", year)
    fishing_path = modeling_path("fishing_training", year)
    species_path = modeling_path("species_training", year)
    h3_grid_path = grid_path()
    seasonal_path = seasonal_lookup_path()

    pred_cols = parquet_columns(con, pred_path)
    feature_cols = parquet_columns(con, feature_path)
    fishing_cols = parquet_columns(con, fishing_path)

    select_cols = []
    select_cols.extend(prediction_select(pred_cols))
    select_cols.extend(feature_select(feature_cols))
    select_cols.extend(
        qualified_columns(fishing_cols, "ft", exclude=set(GRID_KEYS))
    )
    select_cols.extend(species_select(species_path))

    select_sql = ",\n            ".join(select_cols)

    return f"""
        SELECT
            {select_sql}
        FROM {parquet_sql(pred_path)} AS p
        LEFT JOIN {parquet_sql(plaus_path)} AS pl
        USING (h3, date, species)
        LEFT JOIN {parquet_sql(feature_path)} AS fg
        USING (h3, date)
        LEFT JOIN {parquet_sql(fishing_path)} AS ft
        USING (h3, date)
        LEFT JOIN {parquet_sql(h3_grid_path)} AS g
        USING (h3)
        LEFT JOIN {parquet_sql(seasonal_path)} AS sl
        ON sl.adjusted_doy = {adjusted_doy_sql("p.date")}
        {species_join(species_path)}
    """


def build_year(con: duckdb.DuckDBPyConnection, year: int) -> Path:
    """Build one year of the Big Beautiful Dataset."""
    missing = missing_required_inputs(year)
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" + "\n".join(str(path) for path in missing)
        )

    pred_rows = count_rows(con, prediction_path(year))
    out_path = output_path(year)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nYear {year}")
    print(f"  prediction base rows: {pred_rows:,}")
    if not modeling_path("species_training", year).exists():
        print("  species_training missing: observation columns will be NULL")

    query = build_query(con, year)
    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(out_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    out_rows = count_rows(con, out_path)
    if out_rows != pred_rows:
        raise ValueError(
            f"Output row count changed for {year}: {out_rows:,} != {pred_rows:,}"
        )

    print(f"  saved: {out_path}")
    print(f"  output rows: {out_rows:,}")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Build one year. Can be passed more than once.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=YEAR_START,
        help="First year to build when --year is not passed.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=YEAR_END,
        help="Last year to build when --year is not passed.",
    )
    return parser.parse_args()


def main() -> int:
    """Build combined dataset partitions."""
    args = parse_args()
    years = args.year or list(range(args.start_year, args.end_year + 1))
    con = duckdb.connect()
    outputs = []

    for year in years:
        pred_path = prediction_path(year)
        if not pred_path.exists():
            print(f"Skipping missing predictions: {pred_path}")
            continue

        outputs.append(build_year(con, year))

    print(f"\nBuilt {len(outputs)} partitions for {OUTPUT_TABLE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
