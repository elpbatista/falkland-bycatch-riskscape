"""Table inspection utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths


DEFAULT_TABLES = {
    "environmental": paths["data"] / "features" / "environmental",
    "static": paths["data"] / "features" / "static" / "static.parquet",
    "fishing_training": paths["data"] / "modeling" / "fishing_training",
    "species_training": paths["data"] / "modeling" / "species_training",
    "predictions": paths["data"] / "modeling" / "predictions",
}


def resolve_table_path(path_or_alias: str | Path) -> Path:
    """Resolve an explicit path or known table alias."""
    value = str(path_or_alias)

    if value in DEFAULT_TABLES:
        path = DEFAULT_TABLES[value]
        if path.is_dir():
            candidates = sorted(path.glob("year=*/part.parquet"))
            if candidates:
                return candidates[-1]
        return path

    return Path(path_or_alias)


def inspect_table(path_or_alias: str | Path, max_rows: int = 5) -> dict:
    """Inspect a parquet table and print a compact schema summary."""
    path = resolve_table_path(path_or_alias)

    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    df = pd.read_parquet(path)

    summary = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    print("Path:", path)
    print("Rows:", summary["rows"])
    print("Columns:")
    for col in summary["columns"]:
        print(f"  - {col}: {summary['dtypes'][col]}")

    if max_rows > 0:
        print()
        print(df.head(max_rows).to_string())

    return summary
