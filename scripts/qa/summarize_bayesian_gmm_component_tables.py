"""Summarize Bayesian/GMM environmental components for report tables."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import modeling_root


COMPONENTS_PATH = (
    paths["data"]
    / "modeling"
    / "metrics"
    / "bayesian_gmm_joint_gmm_components.csv"
)
OUTPUT_ROOT = paths["data"] / "plot_exports" / "plausibility"
SPECIES_YEARS = [2022, 2023]


def component_summary() -> pd.DataFrame:
    """Return report-ready environmental component summary statistics."""
    components = pd.read_csv(COMPONENTS_PATH)

    out = pd.DataFrame(
        {
            "component": components["component"].astype(int),
            "weight": components["weight"],
            "sst_c": components["sst"] - 273.15,
            "ssh_m": components["ssh"],
            "wind_speed_m_s": components["wind_speed"],
            "chl_mg_m3": np.expm1(components["chl_log"]),
            "depth_m": components["depth_m"],
            "dist_coast_km": components["dist_coast_m"] / 1000.0,
        }
    )

    out = out.sort_values("component").reset_index(drop=True)
    return out.round(
        {
            "weight": 3,
            "sst_c": 2,
            "ssh_m": 3,
            "wind_speed_m_s": 2,
            "chl_mg_m3": 2,
            "depth_m": 0,
            "dist_coast_km": 1,
        }
    )


def species_training_path(year: int) -> Path:
    """Return species-training partition path."""
    return modeling_root("species_training") / f"year={year}" / "part.parquet"


def component_assignment_path(year: int) -> Path:
    """Return component-assignment partition path."""
    return modeling_root("environmental_regimes") / f"year={year}" / "part.parquet"


def load_observed_species_components() -> pd.DataFrame:
    """Join positive observed species-use rows to component assignments."""
    frames: list[pd.DataFrame] = []

    for year in SPECIES_YEARS:
        species_path = species_training_path(year)
        component_path = component_assignment_path(year)

        if not species_path.exists() or not component_path.exists():
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

        components = pd.read_parquet(
            component_path,
            columns=[
                "h3",
                "date",
                "bayesian_gmm_k30_component",
                "bayesian_gmm_k30_component_probability",
            ],
        )
        components = components.rename(
            columns={
                "bayesian_gmm_k30_component": "component",
                "bayesian_gmm_k30_component_probability": "component_probability",
            }
        )

        joined = observed.merge(
            components,
            on=["h3", "date"],
            how="inner",
        )
        joined["year"] = year
        frames.append(joined)

    if not frames:
        raise FileNotFoundError("No observed species/component rows found")

    return pd.concat(frames, ignore_index=True)


def species_component_summary() -> pd.DataFrame:
    """Return component occupancy summary for observed species-use rows."""
    observed = load_observed_species_components()

    summary = (
        observed.groupby(["species", "component"], as_index=False)
        .agg(
            observed_rows=("component", "size"),
            residence_index_sum=("residence_index", "sum"),
            mean_residence_index=("residence_index", "mean"),
            mean_component_probability=("component_probability", "mean"),
            individual_count_sum=("individual_count", "sum"),
            trip_count_sum=("trip_count", "sum"),
        )
    )
    summary["component"] = summary["component"].astype(int)

    totals = summary.groupby("species")["observed_rows"].transform("sum")
    summary["observed_row_percent"] = 100.0 * summary["observed_rows"] / totals

    return summary.sort_values(["species", "component"]).round(
        {
            "residence_index_sum": 2,
            "mean_residence_index": 3,
            "mean_component_probability": 3,
            "observed_row_percent": 1,
        }
    )


def main() -> int:
    """Write report tables."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    component_file = OUTPUT_ROOT / "bayesian_gmm_component_summary.csv"
    component_summary().to_csv(component_file, index=False)
    print("Saved:", component_file)

    species_file = OUTPUT_ROOT / "observed_species_components.csv"
    species_component_summary().to_csv(species_file, index=False)
    print("Saved:", species_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
