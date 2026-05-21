"""Lightweight validation checks for generated workflow products."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskscape.config import paths


VALIDATION_TARGETS = {
    "grid": paths["grids"],
    "environmental_features": paths["data"] / "features" / "environmental",
    "static_features": paths["data"] / "features" / "static" / "static.parquet",
    "fishing_training": paths["data"] / "modeling" / "fishing_training",
    "species_training": paths["data"] / "modeling" / "species_training",
}


def latest_partition(root: Path) -> Path | None:
    """Return the latest partition file under a partitioned table root."""
    candidates = sorted(root.glob("year=*/part.parquet"))
    if not candidates:
        return None
    return candidates[-1]


def representative_file(path: Path) -> Path | None:
    """Return a representative file to inspect for a validation target."""
    if path.is_file():
        return path

    if path.is_dir():
        partition = latest_partition(path)
        if partition:
            return partition

        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return parquet_files[-1]

    return None


def validate_parquet_file(path: Path) -> dict:
    """Return lightweight validation details for one parquet file."""
    df = pd.read_parquet(path)
    result = {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "empty": bool(df.empty),
    }

    if "h3" in df.columns:
        result["unique_h3"] = int(df["h3"].nunique())

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], utc=True)
        result["date_min"] = str(dates.min())
        result["date_max"] = str(dates.max())
        result["unique_dates"] = int(dates.nunique())

    return result


def validate_target(name: str, path: Path) -> dict:
    """Validate one configured target."""
    file_path = representative_file(path)

    if file_path is None:
        return {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "ready": False,
            "reason": "No representative parquet file found",
        }

    if file_path.suffix != ".parquet":
        return {
            "name": name,
            "path": str(file_path),
            "exists": file_path.exists(),
            "ready": file_path.exists(),
            "reason": "Exists but is not a parquet table",
        }

    result = validate_parquet_file(file_path)
    result["name"] = name
    result["ready"] = result["exists"] and not result["empty"]
    return result


def run_quick_validation(
    targets: dict[str, Path] | None = None,
) -> list[dict]:
    """Run lightweight validation checks and print a summary."""
    selected_targets = targets or VALIDATION_TARGETS
    results = [
        validate_target(name, path)
        for name, path in selected_targets.items()
    ]

    for result in results:
        status = "OK" if result["ready"] else "MISSING"
        print(f"[{status}] {result['name']}: {result['path']}")

        if "rows" in result:
            print(f"      rows={result['rows']} columns={len(result['columns'])}")
        if "reason" in result:
            print(f"      {result['reason']}")

    failed = [result for result in results if not result["ready"]]
    if failed:
        raise RuntimeError(f"{len(failed)} validation target(s) are not ready")

    return results
