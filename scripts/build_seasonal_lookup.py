"""Build the seasonal lookup table."""

import numpy as np
import pandas as pd

from riskscape.config import paths


def compute_adjusted_doy(date_series):
    """Return adjusted DOY using the leap-year correction."""
    date_series = pd.to_datetime(date_series)

    doy = date_series.dt.dayofyear
    is_leap = date_series.dt.is_leap_year

    adjusted_doy = doy.copy()
    adjusted_doy[(is_leap) & (doy > 59)] -= 1

    return adjusted_doy.astype("int16")


def build_seasonal_lookup():
    """Build the 365-day seasonal lookup."""
    adjusted_doy = np.arange(1, 366, dtype=np.int16)
    angle = 2 * np.pi * adjusted_doy / 365.0

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

    print(f"Saved seasonal lookup: {out_file}")
    print(df.head())


def main():
    """Run seasonal lookup generation."""
    print("Building seasonal lookup...")
    build_seasonal_lookup()
    print("Seasonal lookup complete.")


if __name__ == "__main__":
    main()