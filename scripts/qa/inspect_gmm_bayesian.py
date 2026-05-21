"""Inspect joint Bayesian GMM ecological regimes."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import numpy as np
import pandas as pd

from riskscape.config import PROJECT_ROOT, paths
from riskscape.indices.seasonal import compute_adjusted_doy

MODEL_NAME = "bayesian_gmm"
YEAR = 2022

MODEL_PATH = (
    paths["data"]
    / "modeling"
    / "models"
    / MODEL_NAME
    / "species_model_joint.joblib"
)

OUT_PATH = (
    paths["data"]
    / "modeling"
    / "metrics"
    / f"{MODEL_NAME}_joint_gmm_components.csv"
)

ASSIGNMENT_SUMMARY_PATH = (
    paths["data"]
    / "modeling"
    / "metrics"
    / f"{MODEL_NAME}_joint_component_assignments_{YEAR}.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / f"{MODEL_NAME}_component_report.txt"
)

COMPONENT_PATH = (
    paths["data"]
    / "modeling"
    / "cube_components"
    / f"year={YEAR}"
    / "part.parquet"
)


def summarize_assignments() -> pd.DataFrame | None:
    """Summarize assigned component coverage."""
    if not COMPONENT_PATH.exists():
        print(f"Skipping assignment summary; missing: {COMPONENT_PATH}")
        return None

    components = pd.read_parquet(
        COMPONENT_PATH,
        columns=[
            "h3",
            "date",
            "species",
            "component",
            "component_probability",
            "component_entropy",
        ],
    )

    components["adjusted_doy"] = compute_adjusted_doy(components["date"])

    grouped = (
        components
        .groupby(["species", "component"], observed=True)
        .agg(
            rows=("component", "size"),
            h3_cells=("h3", "nunique"),
            dates=("date", "nunique"),
            component_probability_mean=("component_probability", "mean"),
            component_entropy_mean=("component_entropy", "mean"),
            adjusted_doy_days=("adjusted_doy", "nunique"),
            adjusted_doy_min=("adjusted_doy", "min"),
            adjusted_doy_max=("adjusted_doy", "max"),
        )
        .reset_index()
    )

    totals = grouped.groupby("species", observed=True)["rows"].transform("sum")
    grouped["species_fraction"] = grouped["rows"] / totals
    grouped = grouped.sort_values(
        ["species", "species_fraction"],
        ascending=[True, False],
    )

    ASSIGNMENT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(ASSIGNMENT_SUMMARY_PATH, index=False)

    print(grouped)
    print(f"Saved: {ASSIGNMENT_SUMMARY_PATH}")

    return grouped


def component_species_labels(components: pd.DataFrame) -> dict[int, str]:
    """Return simple species-dominance labels for components."""
    labels = {}

    for _, row in components.iterrows():
        component = int(row["component"])
        bbal = float(row["species_BBAL"])
        safs = float(row["species_SAFS"])

        if bbal >= 0.8 and safs <= 0.2:
            label = "BBAL-dominated"
        elif safs >= 0.8 and bbal <= 0.2:
            label = "SAFS-dominated"
        else:
            label = "mixed species"

        labels[component] = label

    return labels


def write_text_report(
    components: pd.DataFrame,
    assignments: pd.DataFrame | None,
) -> None:
    """Write a human-readable component report."""
    labels = component_species_labels(components)
    lines = [
        "Bayesian GMM Component Report",
        "================================",
        "",
        "Model",
        "-----",
        "Model: joint Bayesian GMM species-use model",
        f"Components: {len(components)}",
        "Component IDs: "
        + ", ".join(str(int(value)) for value in sorted(components["component"])),
        "",
        "How to read this report",
        "-----------------------",
        "Each component is an environmental regime learned by the joint Bayesian GMM.",
        "The model was trained with species as a one-hot input, so some components are",
        "species-specific and others are shared or transitional. Values close to 1 in",
        "species_BBAL or species_SAFS indicate which species dominates the component.",
        "Tiny negative values such as -0.0000 are numerical noise and should be read as zero.",
        "",
        "The assignment summary is based on actual component assignments for 2022.",
        "It reports coverage and assignment confidence, not mean latitude/longitude or",
        "mean/median DOY. A mean coordinate over a large H3 region and a mean seasonal",
        "day are not meaningful summaries for these components.",
        "",
        "Model component parameters",
        "--------------------------",
    ]

    for _, row in components.sort_values("weight", ascending=False).iterrows():
        component = int(row["component"])
        lines.append(
            f"Component {component}: weight={row['weight']:.3f}, "
            f"{labels[component]}, "
            f"species_BBAL={row['species_BBAL']:.3f}, "
            f"species_SAFS={row['species_SAFS']:.3f}, "
            f"sst={row['sst']:.2f}, "
            f"wind_speed={row['wind_speed']:.2f}, "
            f"chl_log={row['chl_log']:.2f}, "
            f"depth_m={row['depth_m']:.1f}, "
            f"dist_coast_m={row['dist_coast_m']:.0f}"
        )

    if assignments is not None:
        lines.extend(
            [
                "",
                f"{YEAR} assigned component patterns",
                "--------------------------------",
            ]
        )

        for species, species_df in assignments.groupby("species", sort=True):
            lines.extend(["", str(species), "-" * len(str(species))])

            species_df = species_df.sort_values(
                "species_fraction",
                ascending=False,
            )

            for _, row in species_df.iterrows():
                component = int(row["component"])
                lines.append(
                    f"Component {component}: "
                    f"{row['species_fraction'] * 100:.1f}% of assigned "
                    f"{species} rows ({int(row['rows']):,} rows), "
                    f"{labels.get(component, 'unknown')}; "
                    f"{int(row['h3_cells']):,} H3 cells, "
                    f"{int(row['dates']):,} dates, "
                    f"{int(row['adjusted_doy_days'])} adjusted DOY values, "
                    f"adjusted DOY span={int(row['adjusted_doy_min'])}-"
                    f"{int(row['adjusted_doy_max'])}, "
                    f"mean assignment probability="
                    f"{row['component_probability_mean']:.3f}, "
                    f"mean entropy={row['component_entropy_mean']:.3f}"
                )

        lines.extend(
            [
                "",
                "Main interpretation",
                "-------------------",
                "Component assignments summarize model regimes, not direct biological",
                "habitat classes. Species-dominated components indicate regimes where",
                "the joint model strongly separated the species-conditioned feature space.",
                "Mixed components indicate environmental regimes shared across species or",
                "not cleanly separated by species identity.",
            ]
        )

    lines.extend(
        [
            "",
            "Source files",
            "------------",
            str(OUT_PATH),
            str(ASSIGNMENT_SUMMARY_PATH),
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved: {REPORT_PATH}")


def main() -> int:
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    features = payload["features"]
    encoder = payload["encoder"]
    species_features = [
        f"species_{species}" for species in encoder.categories_[0]
    ]
    columns = species_features + features

    means_scaled = model.gmm.means_
    means_raw = model.scaler.inverse_transform(means_scaled)

    std_scaled = np.sqrt(
        np.array([np.diag(cov) for cov in model.gmm.covariances_])
    )

    std_raw = std_scaled * model.scaler.scale_

    means = pd.DataFrame(means_raw, columns=columns)
    std = pd.DataFrame(std_raw, columns=[f"{col}_std" for col in columns])

    out = pd.concat([means, std], axis=1)
    out.insert(0, "component", range(len(out)))
    out.insert(1, "weight", model.gmm.weights_)

    out = out.sort_values("weight", ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(out)
    print(f"Saved: {OUT_PATH}")
    assignments = summarize_assignments()
    write_text_report(out, assignments)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
