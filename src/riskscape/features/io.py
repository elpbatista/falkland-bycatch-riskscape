"""Feature table IO utilities."""

from pathlib import Path

import pandas as pd


def write_year_partition(
    df: pd.DataFrame,
    out_root: Path,
    year: int,
    filename: str = "part.parquet",
) -> Path:
    """Write a year-partitioned ZSTD parquet file."""
    out_dir = out_root / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / filename
    df.to_parquet(
        out_file,
        index=False,
        compression="zstd",
    )

    return out_file