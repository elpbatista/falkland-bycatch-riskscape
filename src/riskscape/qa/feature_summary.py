"""Summarize yearly feature table quality."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from riskscape.config import paths


FEATURE_TABLES = {
    "environmental": paths["data"] / "features" / "environmental",
    "fishing_effort": paths["data"] / "features" / "fishing_effort",
    "species_presence": paths["data"] / "features" / "species_presence",
    "static": paths["data"] / "features" / "static",
}


def numeric_summary(series: pd.Series) -> dict:
    """Return numeric summary for one column."""
    return {
        "missing_fraction": float(series.isna().mean()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def summarize_table(path: Path, table_name: str, year: int | str) -> dict:
    """Return QA summary for one feature table."""
    summary = {
        "table": table_name,
        "year": year,
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return summary

    df = pd.read_parquet(path)

    summary["rows"] = int(len(df))
    summary["columns"] = list(df.columns)

    if "h3" in df.columns:
        summary["unique_h3"] = int(df["h3"].nunique())

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], utc=True)
        summary["unique_dates"] = int(dates.nunique())
        summary["date_min"] = str(dates.min())
        summary["date_max"] = str(dates.max())

    if {"h3", "date"}.issubset(df.columns):
        key_cols = ["h3", "date"]

        if "species" in df.columns:
            key_cols.append("species")

        summary["key_columns"] = key_cols
        summary["duplicate_key_rows"] = int(
            df.duplicated(subset=key_cols).sum()
        )

    if "species" in df.columns:
        summary["species"] = sorted(
            str(value) for value in df["species"].dropna().unique()
        )

    summary["missing_fraction"] = {
        col: float(df[col].isna().mean())
        for col in df.columns
    }

    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in ("h3", "date")
    ]

    summary["numeric"] = {
        col: numeric_summary(df[col])
        for col in numeric_cols
    }

    return summary


def summarize_feature_tables() -> dict:
    """Return summaries for configured feature tables."""
    summaries = {}

    for table_name, root in FEATURE_TABLES.items():
        table_summaries = []

        if not root.exists():
            summaries[table_name] = {
                "exists": False,
                "root": str(root),
                "years": [],
            }
            continue

        if table_name == "static":
            path = root / "static.parquet"
            table_summaries.append(summarize_table(path, table_name, "static"))
        else:
            for year_dir in sorted(root.glob("year=*")):
                year = int(year_dir.name.split("=")[1])
                path = year_dir / "part.parquet"
                table_summaries.append(summarize_table(path, table_name, year))

        summaries[table_name] = {
            "exists": True,
            "root": str(root),
            "years": table_summaries,
        }

    return summaries


def save_feature_qa_summary(summary: dict) -> Path:
    """Save a feature QA summary and return its path."""
    out_file = paths["processed"] / "feature_qa_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_file


def run_feature_qa_summary() -> Path:
    """Run feature QA summary and print the result."""
    summary = summarize_feature_tables()
    out_file = save_feature_qa_summary(summary)

    print(json.dumps(summary, indent=2))
    print()
    print("Saved:", out_file)

    return out_file
