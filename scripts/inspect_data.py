"""Validate standardized variables across full model_ready dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import cfg, paths


Z_COLUMNS = [
    "sst_anom_z",
    "chl_anom_z",
    "ssh_anom_z",
    "sst_grad_z",
    "chl_grad_z",
    "ssh_grad_z",
    "wind_z",
    "wind_anom_z",
    "wind_grad_z",
]


def year_range():
    """Return inclusive year range from config."""
    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def summarize(x):
    """Return summary stats for a numeric array."""
    finite = x[np.isfinite(x)]

    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def main():
    """Validate full-period standardized variables."""
    base = Path(paths["data"]) / "model_ready"

    frames = []

    for year in year_range():
        path = base / f"year={year}.parquet"
        df = pd.read_parquet(path, columns=Z_COLUMNS)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    for col in Z_COLUMNS:
        print(f"\n{col} summary")
        stats = summarize(df[col].to_numpy(dtype=np.float64))
        print(stats)
        print("mean ~ 0:", abs(stats["mean"]) < 0.01)
        print("std ~ 1:", abs(stats["std"] - 1.0) < 0.01)
        print("nan count:", int(df[col].isna().sum()))


if __name__ == "__main__":
    main()

# ------------------------------------------------------------

# """Validate standardized variables in model_ready dataset."""

# from pathlib import Path

# import numpy as np
# import pandas as pd

# from riskscape.config import cfg, paths


# Z_COLUMNS = [
#     "sst_anom_z",
#     "chl_anom_z",
#     "ssh_anom_z",
#     "sst_grad_z",
#     "chl_grad_z",
#     "ssh_grad_z",
#     "wind_z",
#     "wind_anom_z",
#     "wind_grad_z",
# ]


# def year_range():
#     """Return inclusive year range from config."""
#     start = pd.to_datetime(cfg["time"]["start"]).year
#     end = pd.to_datetime(cfg["time"]["end"]).year
#     return range(start, end + 1)


# def summarize(series):
#     """Return summary statistics for a pandas Series."""
#     x = series.to_numpy(dtype=np.float64)
#     finite = x[np.isfinite(x)]

#     if finite.size == 0:
#         return {"count": 0}

#     return {
#         "count": int(finite.size),
#         "mean": float(np.mean(finite)),
#         "std": float(np.std(finite)),
#         "p01": float(np.percentile(finite, 1)),
#         "p50": float(np.percentile(finite, 50)),
#         "p99": float(np.percentile(finite, 99)),
#         "min": float(np.min(finite)),
#         "max": float(np.max(finite)),
#     }


# def validate_year(year):
#     """Validate standardized variables for one year."""
#     path = Path(paths["data"]) / "model_ready" / f"year={year}.parquet"

#     print(f"\n=== {path.name} ===")

#     df = pd.read_parquet(path, columns=Z_COLUMNS)

#     for col in Z_COLUMNS:
#         print(f"\n{col} summary")

#         stats = summarize(df[col])
#         print(stats)

#         if stats["count"] > 0:
#             mean_ok = abs(stats["mean"]) < 0.05
#             std_ok = abs(stats["std"] - 1.0) < 0.05

#             print("mean ~ 0:", mean_ok)
#             print("std ~ 1:", std_ok)

#             nan_count = df[col].isna().sum()
#             print("nan count:", int(nan_count))


# def main():
#     """Run validation across all years."""
#     for year in year_range():
#         validate_year(year)


# if __name__ == "__main__":
#     main()

# ------------------------------------------------------

# """Sanity check CHL handling in model_ready tables."""

# from pathlib import Path

# import numpy as np
# import pandas as pd

# from riskscape.config import cfg, paths


# def year_range():
#     """Return inclusive year range from config."""

#     start = pd.to_datetime(cfg["time"]["start"]).year
#     end = pd.to_datetime(cfg["time"]["end"]).year
#     return range(start, end + 1)


# def summarize(series):
#     """Return compact summary for a numeric series."""

#     finite = series[np.isfinite(series)]

#     if len(finite) == 0:
#         return {
#             "count": 0,
#             "min": np.nan,
#             "p01": np.nan,
#             "p50": np.nan,
#             "p99": np.nan,
#             "max": np.nan,
#             "mean": np.nan,
#         }

#     return {
#         "count": len(finite),
#         "min": float(np.min(finite)),
#         "p01": float(np.percentile(finite, 1)),
#         "p50": float(np.percentile(finite, 50)),
#         "p99": float(np.percentile(finite, 99)),
#         "max": float(np.max(finite)),
#         "mean": float(np.mean(finite)),
#     }


# def inspect_year(year):
#     """Inspect one model_ready yearly table."""

#     path = Path(paths["data"]) / "model_ready" / f"year={year}.parquet"
#     if not path.exists():
#         raise FileNotFoundError(f"Missing file: {path}")

#     df = pd.read_parquet(
#         path,
#         columns=["chl", "chl_grad", "chl_anom"],
#     )

#     chl = df["chl"].to_numpy(dtype=np.float64)
#     chl_grad = df["chl_grad"].to_numpy(dtype=np.float64)
#     chl_anom = df["chl_anom"].to_numpy(dtype=np.float64)

#     print(f"\n=== {year} ===")
#     print(f"File: {path}")
#     print(f"Rows: {len(df):,}")

#     print("\nchl summary")
#     print(summarize(chl))
#     print("chl negative values:", int(np.sum(chl < 0)))
#     print("chl values > 10:", int(np.sum(chl > 10)))

#     print("\nchl_grad summary")
#     print(summarize(chl_grad))
#     print("chl_grad negative values:", int(np.sum(chl_grad < 0)))

#     print("\nchl_anom summary")
#     print(summarize(chl_anom))
#     print("chl_anom negative values:", int(np.sum(chl_anom < 0)))
#     print("chl_anom positive values:", int(np.sum(chl_anom > 0)))

#     print("\nChecks")
#     print(
#         "chl looks raw:",
#         bool(np.sum(chl < 0) == 0 and np.nanmax(chl) > 10),
#     )
#     print(
#         "chl_grad looks non-negative:",
#         bool(np.nanmin(chl_grad) >= 0),
#     )
#     print(
#         "chl_anom looks centered around zero:",
#         bool(abs(np.nanmean(chl_anom)) < 0.1),
#     )


# def main():
#     """Run checks for all years."""

#     for year in year_range():
#         inspect_year(year)


# if __name__ == "__main__":
#     main()

# -------------------------------------------------------

# """Quick inspection of model_ready dataset."""

# import pandas as pd


# def main():
#     path = "data/model_ready/year=2019.parquet"

#     print(f"Loading: {path}")
#     df = pd.read_parquet(path)

#     print("\nColumns:")
#     print(list(df.columns))

#     print("\nShape:")
#     print(df.shape)

#     print("\nHead:")
#     print(df.head())


# if __name__ == "__main__":
#     main()

# -------------------------------------------------------

# """Check whether CHL is raw or log-transformed."""

# import numpy as np
# import pandas as pd


# def main():
#     path = "data/model_ready/year=2019.parquet"

#     df = pd.read_parquet(path)

#     chl = df["chl"].to_numpy()

#     finite = chl[np.isfinite(chl)]

#     print("\nBasic stats:")
#     print("min:", finite.min())
#     print("max:", finite.max())
#     print("mean:", finite.mean())

#     print("\nPercentiles:")
#     print(np.percentile(finite, [1, 5, 25, 50, 75, 95, 99]))

#     print("\nNegative values count:", (finite < 0).sum())
#     print("Values > 10 count:", (finite > 10).sum())

#     print("\nSample values:")
#     print(finite[:10])


# if __name__ == "__main__":
#     main()

# ----------------------------------------------------------