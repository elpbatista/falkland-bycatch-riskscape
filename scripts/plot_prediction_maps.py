"""Plot prediction maps from model prediction and plausibility outputs."""

from __future__ import annotations

from dataclasses import replace

from riskscape.config import paths
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    MapStyle,
    plot_hazard_map,
    plot_hazard_plausibility_map,
    plot_prediction_map,
)


YEAR = 2022
MODEL_NAMES = ["hybrid_presence_gate_extra_trees_bayesian_gmm"]
PREDICTION_PRODUCTS = [
    ("joint", "BBAL"),
    ("joint", "SAFS"),
]
AGG = "non_zero_median"
PLAUSIBILITY_AGG = "non_zero_median"
PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD = 0.25


RISK_STYLE = MapStyle(
    color_scale="log",
    alpha_scale=False,
    show_reference_map=False,
    min_display_value=MINIMUM_EFFORT_UNIT,
    color_min=MINIMUM_EFFORT_UNIT,
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
)

REALIZED_RISK_STYLE = replace(
    RISK_STYLE,
    title="Realized Risk",
    colorbar_title="Realized Risk",
)
HAZARD_STYLE = replace(
    RISK_STYLE,
    title="Latent Minimum Risk",
    colorbar_title="Latent Minimum Risk",
)
HAZARD_PLAUSIBILITY_STYLE = replace(
    RISK_STYLE,
    title="Latent Minimum Plausible Risk",
    colorbar_title="Latent Minimum Plausible Risk",
)

SPECIES_USE_STYLE = MapStyle(
    title="Species Use",
    colorbar_title="Species Use",
    show_reference_map=False,
    min_display_value=0.0,
)


def prediction_path(
    year: int,
    model_name: str,
    product_name: str,
):
    """Return one prediction partition path."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / model_name
        / product_name
        / f"year={year}"
        / "part.parquet"
    )


def main() -> int:
    """Run prediction map plots."""
    for model_name in MODEL_NAMES:
        for product_name, species in PREDICTION_PRODUCTS:
            input_path = prediction_path(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
            )

            if not input_path.exists():
                print(f"Skipping missing input: {input_path}")
                continue

            print(f"Input: {input_path}")

            out_file = plot_prediction_map(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
                value_col="species_use_log_pred",
                species=species,
                agg=AGG,
                style=SPECIES_USE_STYLE,
            )
            print(f"Saved: {out_file}")

            out_file = plot_prediction_map(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
                value_col="risk_log_pred",
                species=species,
                agg=AGG,
                style=REALIZED_RISK_STYLE,
            )
            print(f"Saved: {out_file}")

            out_file = plot_hazard_map(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
                species=species,
                agg=AGG,
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                style=HAZARD_STYLE,
            )
            print(f"Saved: {out_file}")

            out_file = plot_hazard_plausibility_map(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
                species=species,
                agg=AGG,
                plausibility_agg=PLAUSIBILITY_AGG,
                confidence_threshold=PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD,
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                style=HAZARD_PLAUSIBILITY_STYLE,
            )
            print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
