# """Inspect yearly feature tables."""

# from pathlib import Path

# import pandas as pd

# from riskscape.config import paths


# def inspect_table(path: Path) -> None:
#     """Print basic table diagnostics."""
#     if not path.exists():
#         raise FileNotFoundError(f"Table not found: {path}")

#     df = pd.read_parquet(path)

#     print("\nTable:", path)
#     print("\nHead:")
#     print(df.head())

#     print("\nDescribe:")
#     print(df.describe())

#     print("\nMissing fraction:")
#     print(df.isna().mean())

#     print("\nShape:")
#     print(df.shape)

#     print("\nColumns:")
#     print(df.columns.tolist())

#     if {"h3", "date"}.issubset(df.columns):
#         duplicate_count = df.duplicated(subset=["h3", "date"]).sum()
#         print("\nDuplicate h3/date rows:")
#         print(duplicate_count)

#         print("\nUnique H3 cells:")
#         print(df["h3"].nunique())

#         print("\nUnique dates:")
#         print(df["date"].nunique())

#         print("\nRows per date:")
#         print(df.groupby("date").size().describe())

#         print("\nDate range:")
#         dates = pd.to_datetime(df["date"], utc=True)
#         print(dates.min(), "->", dates.max())

#     if "fishing_hours" in df.columns:
#         print("\nFishing effort zero rows:")
#         print((df["fishing_hours"] == 0).sum())

#         print("\nFishing hours quantiles:")
#         print(df["fishing_hours"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]))

#     if "vessel_count" in df.columns:
#         print("\nVessel count quantiles:")
#         print(df["vessel_count"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]))


# def main() -> int:
#     """Run table inspection."""
#     year = 2014

#     tables = [
#         paths["data"] / "features" / "environmental" / f"year={year}" / "part.parquet",
#         paths["data"] / "features" / "fishing_effort" / f"year={year}" / "part.parquet",
#     ]

#     for path in tables:
#         inspect_table(path)

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

import pandas as pd

from riskscape.config import paths

df = pd.read_parquet(

    paths["raw"] / "gfw" / "year=2014" / "fishing_effort.parquet"

)

dates = pd.to_datetime(df["date"], utc=True)

print(dates.min(), dates.max())

print(dates.dt.floor("D").nunique())

dates = pd.to_datetime(df["date"], utc=True)

print(dates.max())

print(dates.dt.floor("D").nunique())