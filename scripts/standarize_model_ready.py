"""Standardize model_ready predictors using full-period statistics."""

from pathlib import Path

import json
import numpy as np
import pandas as pd

from riskscape.config import cfg, paths


COLUMNS = [
    "sst",
    "chl",
    "ssh",
    "sst_grad",
    "chl_grad",
    "ssh_grad",
    "sst_anom",
    "chl_anom",
    "ssh_anom",
    "wind",
    "wind_anom",
    "wind_grad",
]


def year_range():
    """Return inclusive year range from config."""
    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def compute_stats():
    """Compute full-period mean and std for selected columns."""
    stats = {col: {"count": 0, "sum": 0.0, "sumsq": 0.0} for col in COLUMNS}
    src_dir = Path(paths["data"]) / "model_ready"

    for year in year_range():
        path = src_dir / f"year={year}.parquet"
        df = pd.read_parquet(path, columns=COLUMNS)

        for col in COLUMNS:
            x = df[col].to_numpy(dtype=np.float64)
            finite = x[np.isfinite(x)]

            stats[col]["count"] += int(finite.size)
            stats[col]["sum"] += float(np.sum(finite))
            stats[col]["sumsq"] += float(np.sum(finite ** 2))

    final = {}

    for col, s in stats.items():
        count = s["count"]

        if count == 0:
            final[col] = {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
            }
            continue

        mean = s["sum"] / count
        var = (s["sumsq"] / count) - (mean ** 2)
        std = np.sqrt(max(var, 0.0))

        final[col] = {
            "mean": float(mean),
            "std": float(std),
            "count": int(count),
        }

    return final


def apply_standardization(stats):
    """Write standardized columns back into model_ready yearly files."""
    src_dir = Path(paths["data"]) / "model_ready"

    for year in year_range():
        path = src_dir / f"year={year}.parquet"
        df = pd.read_parquet(path)

        for col in COLUMNS:
            mean = stats[col]["mean"]
            std = stats[col]["std"]
            z_col = f"{col}_z"

            if np.isfinite(std) and std > 0:
                df[z_col] = ((df[col] - mean) / std).astype("float32")
            else:
                df[z_col] = np.nan

        df.to_parquet(path, index=False)

        print(f"Updated: {path}")
        print(f"Rows: {len(df)}")


def save_stats(stats):
    """Save scaling parameters to disk."""
    out_dir = Path(paths["data"]) / "model_ready"
    out_path = out_dir / "standardization_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats: {out_path}")


def main():
    """Compute and apply standardization."""
    print("Computing full-period statistics")
    stats = compute_stats()

    print("Saving statistics")
    save_stats(stats)

    print("Applying standardization")
    apply_standardization(stats)


if __name__ == "__main__":
    main()