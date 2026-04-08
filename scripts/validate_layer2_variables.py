"""Validate Layer 2 variables in model_ready."""

from pathlib import Path

import numpy as np
import pandas as pd


VARS = [
    "sst",
    "sst_grad",
    "sst_anom",
    "chl",
    "chl_grad",
    "chl_anom",
    "ssh",
    "ssh_grad",
    "ssh_anom",
]


def summarize(x):
    """Return robust summary stats."""
    finite = x[np.isfinite(x)]

    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def validate_year(path):
    """Validate one year file."""
    df = pd.read_parquet(path, columns=VARS)

    print(f"\n=== {path.name} ===")
    print(f"Rows: {len(df)}")

    for var in VARS:
        x = df[var].to_numpy(dtype=np.float64)
        finite = x[np.isfinite(x)]

        print(f"\n{var} summary")
        print(summarize(x))

        print(f"{var} negative:", int(np.sum(finite < 0)))
        print(f"{var} positive:", int(np.sum(finite > 0)))

        if var.endswith("_grad"):
            print(f"{var} should be >= 0:", bool(np.all(finite >= 0)))

        if var.endswith("_anom"):
            print(f"{var} mean ~ 0:", float(np.mean(finite)))


def main():
    """Run validation for all years."""
    base = Path("data/model_ready")

    for year in range(2014, 2024):
        path = base / f"year={year}.parquet"
        validate_year(path)


if __name__ == "__main__":
    main()