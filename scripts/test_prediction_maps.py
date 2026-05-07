"""Plot species prediction maps."""

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

RISK_STYLE = MapStyle(
    color_scale="log",
    alpha_scale=False,
    show_reference_map=False,
    min_display_value=MINIMUM_EFFORT_UNIT,
    color_min=MINIMUM_EFFORT_UNIT,
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
)

REALIZED_RISK_STYLE = replace(RISK_STYLE, title="Realized Risk")
HAZARD_STYLE = replace(RISK_STYLE, title="Latent Minimum Risk")

HAZARD_PLAUSIBILITY_STYLE = MapStyle(
    title="Latent Minimum Plausible Risk",
    color_scale="log",
    alpha_scale=False,
    alpha_col="plausibility_non_zero_median",
    show_reference_map=False,
    min_display_value=MINIMUM_EFFORT_UNIT,
    color_min=MINIMUM_EFFORT_UNIT,
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
)

SPECIES_USE_STYLE = MapStyle(
    title="Species Use",
    show_reference_map=False,
    min_display_value=0.0,
)


def main() -> int:
    """Run prediction map plots."""
    year = 2022
    confidence_threshold = 0.25
    model_names = [
        # "extra_trees",
        "hybrid_presence_gate_extra_trees_bayesian_gmm",
        # "bayesian_gmm",
    ]

    prediction_products = [
        ("bbal", "BBAL"),
        ("safs", "SAFS"),
        ("joint", "BBAL"),
        ("joint", "SAFS"),
    ]

    for model_name in model_names:
        for product_name, species in prediction_products:
            input_path = (
                paths["data"]
                / "modeling"
                / "predictions"
                / model_name
                / product_name
                / f"year={year}"
                / "part.parquet"
            )

            if not input_path.exists():
                print(f"Skipping missing input: {input_path}")
                continue

            print(f"Input: {input_path}")

            plot_prediction_map(
                year=year,
                model_name=model_name,
                product_name=product_name,
                value_col="species_use_log_pred",
                species=species,
                agg="non_zero_median",
                style=SPECIES_USE_STYLE,
            )

            plot_prediction_map(
                year=year,
                model_name=model_name,
                product_name=product_name,
                value_col="risk_log_pred",
                species=species,
                agg="non_zero_median",
                style=REALIZED_RISK_STYLE,
            )

            plot_hazard_map(
                year=year,
                model_name=model_name,
                product_name=product_name,
                species=species,
                agg="non_zero_median",
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                style=HAZARD_STYLE,
            )

            plot_hazard_plausibility_map(
                year=year,
                model_name=model_name,
                product_name=product_name,
                species=species,
                agg="non_zero_median",
                plausibility_agg="non_zero_median",
                confidence_threshold=confidence_threshold,
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                style=HAZARD_PLAUSIBILITY_STYLE,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
