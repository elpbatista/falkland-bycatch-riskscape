"""Environmental anomalies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths


def load_partitioned(table: str) -> list[tuple[int, Path]]:
    root = paths["data"] / "features" / table
    parts = []

    for path in sorted(root.glob("year=*/part.parquet")):
        year = int(path.parent.name.split("=")[1])
        parts.append((year, path))

    if not parts:
        raise FileNotFoundError(f"No data for table: {table}")

    return parts


def load_all(parts: list[tuple[int, Path]]) -> pd.DataFrame:
    frames = []

    for _, path in parts:
        frames.append(pd.read_parquet(path))

    return pd.concat(frames, ignore_index=True)


def add_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "h3",
        "adjusted_doy",
        "sst",
        "chl_log",
        "ssh",
        "wind_speed",
    }

    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    variables = ["sst", "chl_log", "ssh", "wind_speed"]

    clim = (
        df.groupby(["h3", "adjusted_doy"], observed=True)[variables]
        .transform("mean")
    )

    for var in variables:
        df[f"{var}_anom"] = (df[var] - clim[var]).astype("float32")

    return df


def write_partitions(df: pd.DataFrame, parts: list[tuple[int, Path]]) -> None:
    for year, path in parts:
        df_year = df[df["date"].dt.year == year]

        df_year.to_parquet(path, index=False, compression="zstd")

        print(f"Updated anomalies year={year}")


def process_environmental_anomalies() -> None:
    parts = load_partitioned("environmental")

    df = load_all(parts)

    df["date"] = pd.to_datetime(df["date"])

    df = add_anomalies(df)

    write_partitions(df, parts)
