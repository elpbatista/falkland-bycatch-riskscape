"""Summarize observed species-use records by KMeans seascape class."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import modeling_root


MODEL_NAME = "kmeans_k10"
SPECIES_YEARS = [2022, 2023]
OUTPUT_ROOT = paths["data"] / "plot_exports" / "seascapes"


def species_training_path(year: int) -> Path:
    """Return species-training partition path."""
    return modeling_root("species_training") / f"year={year}" / "part.parquet"


def seascape_assignment_path(year: int, model_name: str = MODEL_NAME) -> Path:
    """Return seascape-assignment partition path."""
    return (
        modeling_root("seascapes")
        / model_name
        / f"year={year}"
        / "part.parquet"
    )


def load_observed_species_seascapes(
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Join positive observed species-use rows to feature-only seascapes."""
    frames: list[pd.DataFrame] = []

    for year in SPECIES_YEARS:
        species_path = species_training_path(year)
        seascape_path = seascape_assignment_path(year, model_name=model_name)

        if not species_path.exists() or not seascape_path.exists():
            continue

        observed = pd.read_parquet(
            species_path,
            columns=[
                "h3",
                "date",
                "species",
                "residence_index",
                "presence_count",
                "individual_count",
                "trip_count",
            ],
        )
        observed = observed[observed["residence_index"] > 0].copy()

        seascapes = pd.read_parquet(
            seascape_path,
            columns=[
                "h3",
                "date",
                "seascape",
                "seascape_distance",
            ],
        )

        joined = observed.merge(
            seascapes,
            on=["h3", "date"],
            how="inner",
        )
        joined["year"] = year
        frames.append(joined)

    if not frames:
        raise FileNotFoundError("No observed species/seascape rows found")

    return pd.concat(frames, ignore_index=True)


def species_seascape_summary(model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Return seascape occupancy summary for observed species-use rows."""
    observed = load_observed_species_seascapes(model_name=model_name)

    summary = (
        observed.groupby(["species", "seascape"], as_index=False)
        .agg(
            observed_rows=("seascape", "size"),
            residence_index_sum=("residence_index", "sum"),
            mean_residence_index=("residence_index", "mean"),
            mean_seascape_distance=("seascape_distance", "mean"),
            individual_count_sum=("individual_count", "sum"),
            trip_count_sum=("trip_count", "sum"),
        )
    )
    summary["seascape"] = summary["seascape"].astype(int)

    totals = summary.groupby("species")["observed_rows"].transform("sum")
    summary["observed_row_percent"] = 100.0 * summary["observed_rows"] / totals

    return summary.sort_values(["species", "seascape"]).round(
        {
            "residence_index_sum": 2,
            "mean_residence_index": 3,
            "mean_seascape_distance": 3,
            "observed_row_percent": 1,
        }
    )


def main() -> int:
    """Write seascape species-use summary tables."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    species_file = OUTPUT_ROOT / f"observed_species_seascapes_{MODEL_NAME}.csv"
    species_seascape_summary(model_name=MODEL_NAME).to_csv(
        species_file,
        index=False,
    )
    print("Saved:", species_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
