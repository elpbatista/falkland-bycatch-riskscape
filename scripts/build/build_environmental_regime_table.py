"""Build a consolidated environmental-regime table.

The output is species-independent and keyed by H3/day:

    h3, date, som_prototype, seascape, seascape_distance,
    bayesian_gmm_k30_component,
    bayesian_gmm_k30_component_probability,
    bayesian_gmm_k30_component_entropy
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

from riskscape.config import paths


SEASCAPE_TABLE = "seascapes/som_15x15_hierarchical_k30"
BAYESIAN_K30_TABLE = "cube_components_bayesian_gmm_k30"
OUTPUT_TABLE = "environmental_regimes"


def table_root(table_name: str) -> Path:
    """Return one modeling table root."""
    return paths["data"] / "modeling" / table_name


def partition_path(table_name: str, year: int) -> Path:
    """Return one yearly partition path."""
    return table_root(table_name) / f"year={year}" / "part.parquet"


def output_path(table_name: str, year: int) -> Path:
    """Return one output partition path."""
    return partition_path(table_name, year)


def available_years(table_name: str) -> list[int]:
    """Return available partition years for a table."""
    years = []
    for path in sorted(table_root(table_name).glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=")[1]))
    return years


def sql_path(path: Path) -> str:
    """Return a SQL-safe string literal."""
    return "'" + str(path).replace("'", "''") + "'"


def remove_existing(path: Path, overwrite: bool) -> None:
    """Remove an existing output partition when allowed."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite.")
        path.unlink()


def build_year(year: int, output_table: str, overwrite: bool) -> Path:
    """Build one consolidated yearly environmental-regime partition."""
    seascape_path = partition_path(SEASCAPE_TABLE, year)
    component_path = partition_path(BAYESIAN_K30_TABLE, year)
    out_path = output_path(output_table, year)

    if not seascape_path.exists():
        raise FileNotFoundError(f"Missing selected seascape partition: {seascape_path}")
    if not component_path.exists():
        raise FileNotFoundError(f"Missing Bayesian/GMM k30 partition: {component_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(out_path, overwrite)

    query = f"""
        COPY (
            SELECT
                CAST(k.h3 AS UBIGINT) AS h3,
                CAST(k.date AS DATE) AS date,
                CAST(k.som_prototype AS SMALLINT) AS som_prototype,
                CAST(k.seascape AS SMALLINT) AS seascape,
                CAST(k.seascape_distance AS FLOAT) AS seascape_distance,
                CAST(c.component AS SMALLINT) AS bayesian_gmm_k30_component,
                CAST(c.component_probability AS FLOAT)
                    AS bayesian_gmm_k30_component_probability,
                CAST(c.component_entropy AS FLOAT)
                    AS bayesian_gmm_k30_component_entropy
            FROM read_parquet({sql_path(seascape_path)}) k
            INNER JOIN read_parquet({sql_path(component_path)}) c
            ON CAST(k.h3 AS UBIGINT) = CAST(c.h3 AS UBIGINT)
            AND CAST(k.date AS DATE) = CAST(c.date AS DATE)
            ORDER BY date, h3
        ) TO {sql_path(out_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    with duckdb.connect(database=":memory:") as con:
        con.execute(query)

    return out_path


def verify_year(year: int, output_table: str) -> dict[str, int]:
    """Return row-count verification for one consolidated partition."""
    seascape_path = partition_path(SEASCAPE_TABLE, year)
    component_path = partition_path(BAYESIAN_K30_TABLE, year)
    out_path = output_path(output_table, year)

    query = f"""
        WITH k AS (
            SELECT CAST(h3 AS UBIGINT) AS h3, CAST(date AS DATE) AS date
            FROM read_parquet({sql_path(seascape_path)})
        ),
        c AS (
            SELECT CAST(h3 AS UBIGINT) AS h3, CAST(date AS DATE) AS date
            FROM read_parquet({sql_path(component_path)})
        ),
        out AS (
            SELECT CAST(h3 AS UBIGINT) AS h3, CAST(date AS DATE) AS date
            FROM read_parquet({sql_path(out_path)})
        )
        SELECT
            (SELECT COUNT(*) FROM k)::BIGINT AS seascape_rows,
            (SELECT COUNT(*) FROM c)::BIGINT AS component_k30_rows,
            (SELECT COUNT(*) FROM out)::BIGINT AS environmental_regime_rows,
            (SELECT COUNT(*) FROM k ANTI JOIN out USING (h3, date))::BIGINT
                AS seascapes_missing_from_output,
            (SELECT COUNT(*) FROM c ANTI JOIN out USING (h3, date))::BIGINT
                AS components_missing_from_output
    """

    with duckdb.connect(database=":memory:") as con:
        row = con.execute(query).fetchone()

    keys = [
        "seascape_rows",
        "component_k30_rows",
        "environmental_regime_rows",
        "seascapes_missing_from_output",
        "components_missing_from_output",
    ]
    return dict(zip(keys, row, strict=True))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build consolidated environmental-regime table.",
    )
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run environmental-regime build."""
    args = parse_args()
    years = args.year or available_years(SEASCAPE_TABLE)

    for year in years:
        out_path = build_year(
            year=year,
            output_table=args.output_table,
            overwrite=args.overwrite,
        )
        check = verify_year(year, args.output_table)
        print(f"Saved: {out_path}")
        print(check)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
