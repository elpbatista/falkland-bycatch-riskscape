"""Date helpers for H3/day analysis keys."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def utc_day(value) -> pd.Timestamp:
    """Return a timezone-free UTC calendar-day timestamp."""
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.floor("D").tz_localize(None)


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``date`` to timezone-free UTC calendar days if present."""
    if "date" not in df.columns:
        return df

    out = df.copy()
    out["date"] = (
        pd.to_datetime(out["date"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    return out


def read_parquet_utc_day(
    path: Path,
    columns: list[str],
    date,
) -> pd.DataFrame:
    """Read rows for one UTC day across timezone-aware or naive Parquet dates."""
    target = utc_day(date)
    filter_values = [
        target,
        target.tz_localize("UTC"),
    ]

    last_error: Exception | None = None
    for value in filter_values:
        try:
            return pd.read_parquet(
                path,
                columns=columns,
                filters=[("date", "=", value)],
            )
        except Exception as exc:  # Parquet engines raise different timestamp errors.
            last_error = exc

    try:
        out = normalize_date_column(pd.read_parquet(path, columns=columns))
    except Exception:
        if last_error is not None:
            raise last_error
        raise

    return out[out["date"].eq(target)].copy()
