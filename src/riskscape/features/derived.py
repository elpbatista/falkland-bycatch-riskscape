"""Derived feature generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.utils.dates import normalize_date_column


def load_partitioned(table: str) -> list[tuple[int, Path]]:
    """Load partition paths for a feature table."""
    root = paths["data"] / "features" / table
    parts = []

    for path in sorted(root.glob("year=*/part.parquet")):
        year = int(path.parent.name.split("=")[1])
        parts.append((year, path))

    if not parts:
        raise FileNotFoundError(f"No data for table: {table}")

    return parts


def compute_adjusted_doy(date_series: pd.Series) -> pd.Series:
    """Return adjusted DOY using leap-year correction."""
    date_series = pd.to_datetime(date_series)

    doy = date_series.dt.dayofyear
    is_leap = date_series.dt.is_leap_year

    adjusted_doy = doy.copy()
    adjusted_doy[(is_leap) & (doy > 59)] -= 1

    return adjusted_doy.astype("int16")


def add_wind_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind speed from u/v components."""
    required = {"wind_u10", "wind_v10"}

    if not required.issubset(df.columns):
        raise KeyError("Missing wind_u10 or wind_v10")

    out = df.copy()
    out["wind_speed"] = np.sqrt(
        out["wind_u10"] ** 2 + out["wind_v10"] ** 2
    ).astype("float32")

    return out


def add_chl_log(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed chlorophyll."""
    if "chl" not in df.columns:
        raise KeyError("Missing chl")

    out = df.copy()
    out["chl_log"] = np.log1p(out["chl"]).astype("float32")

    return out


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted DOY and cyclic seasonal features."""
    if "date" not in df.columns:
        raise KeyError("Missing date")

    out = df.copy()
    adjusted_doy = compute_adjusted_doy(out["date"])

    angle = 2.0 * np.pi * (adjusted_doy - 1) / 365.0

    out["adjusted_doy"] = adjusted_doy
    out["doy_sin"] = np.sin(angle).astype("float32")
    out["doy_cos"] = np.cos(angle).astype("float32")

    return out


def process_environmental() -> None:
    """Add derived variables to environmental feature partitions."""
    parts = load_partitioned("environmental")

    for year, path in parts:
        df = normalize_date_column(pd.read_parquet(path))

        df = add_wind_speed(df)
        df = add_chl_log(df)
        df = add_seasonal_features(df)

        df = normalize_date_column(df)
        df.to_parquet(path, index=False, compression="zstd")

        print(f"Updated environmental year={year}")
        # print(df.head())
