"""Date helpers for H3/day analysis keys."""

from __future__ import annotations

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
