"""Generic sanity check for Layer 2 tables."""

from pathlib import Path

import pandas as pd

from riskscape.config import cfg, paths


def numeric_summary(df):
    """Return describe for numeric columns."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return None
    return num.describe().T


def null_summary(df):
    """Return null counts and percentages."""
    nulls = df.isna().sum()
    pct = (nulls / len(df)) * 100
    out = pd.DataFrame({
        "nulls": nulls,
        "pct": pct
    })
    return out[out["nulls"] > 0].sort_values("pct", ascending=False)


def constant_columns(df):
    """Detect columns with a single unique value."""
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def main():

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    for year in range(start_year, end_year + 1):

        print(f"\n=== {year} ===")

        file_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"

        if not file_path.exists():
            print("Missing file")
            continue

        df = pd.read_parquet(file_path)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nColumn types:")
        print(df.dtypes)

        print("\nNumeric summary:")
        summary = numeric_summary(df)
        if summary is not None:
            print(summary[["mean", "std", "min", "max"]])
        else:
            print("No numeric columns")

        print("\nNulls (>0 only):")
        nulls = null_summary(df)
        if nulls.empty:
            print("None")
        else:
            print(nulls)

        print("\nConstant columns:")
        const = constant_columns(df)
        if const:
            print(const)
        else:
            print("None")

        print("\nBasic checks:")

        if "h3" in df.columns:
            print("h3 unique:", df["h3"].nunique())

        if "date" in df.columns:
            print("date range:", df["date"].min(), "→", df["date"].max())

        print("-" * 40)


if __name__ == "__main__":
    main()