"""Shared utilities for download modules."""

import pandas as pd


def build_year_ranges(start_date: str, end_date: str):
    """Yield yearly date ranges between start and end."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)

        chunk_start = max(start, year_start)
        chunk_end = min(end, year_end)

        if chunk_start <= chunk_end:
            yield (
                year,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )