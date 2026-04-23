"""Build the seasonal lookup table."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import paths


logger = logging.getLogger(__name__)


def compute_adjusted_doy(date_series: pd.Series) -> pd.Series:
    """Return adjusted DOY using leap-year correction."""
    date_series = pd.to_datetime(date_series)

    doy = date_series.dt.dayofyear
    is_leap = date_series.dt.is_leap_year

    adjusted_doy = doy.copy()
    adjusted_doy[(is_leap) & (doy > 59)] -= 1

    return adjusted_doy.astype("int16")


def build_seasonal_lookup() -> Path:
    """Build and save the 365-day seasonal lookup table."""
    adjusted_doy = np.arange(1, 366, dtype=np.int16)
    angle = 2.0 * np.pi * adjusted_doy / 365.0

    df = pd.DataFrame(
        {
            "adjusted_doy": adjusted_doy,
            "doy_sin": np.sin(angle).astype("float32"),
            "doy_cos": np.cos(angle).astype("float32"),
        }
    )

    out_dir = paths["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "seasonal_lookup.parquet"
    df.to_parquet(out_file, index=False)

    logger.info("Seasonal lookup saved: %s", out_file)
    logger.info("Rows: %d", len(df))

    # print(df.head())

    return out_file