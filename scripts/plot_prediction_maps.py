"""Build prediction maps from the Big Beautiful Dataset."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from riskscape.config import paths
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    MapStyle,
    aggregation_name,
    figure_root,
    plot_hazard_df_map,
    plot_hazard_plausibility_df_map,
    plot_prediction_df_map,
)


YEAR = 2022
DATASET_NAME = "big_beautiful_dataset"
AGG = "non_zero_median"
PLAUSIBILITY_AGG = "non_zero_median"
PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD = 0.25
SPECIES = ["BBAL", "SAFS"]

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


def input_path(year: int = YEAR) -> Path:
    """Return one Big Beautiful Dataset partition path."""
    return (
        paths["data"]
        / "modeling"
        / DATASET_NAME
        / f"year={year}"
        / "part.parquet"
    )


def load_bbd(year: int = YEAR) -> pd.DataFrame:
    """Load BBD columns needed for prediction maps."""
    path = input_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Big Beautiful Dataset not found: {path}")

    return pd.read_parquet(
        path,
        columns=[
            "h3",
            "species",
            "species_use_log_pred",
            "risk_log_pred",
            "plausibility",
        ],
    )


def output_file(
    value_name: str,
    species: str,
    year: int,
    suffix: str = "",
) -> Path:
    """Return an output map path."""
    suffix = f"_{suffix}" if suffix else ""

    return (
        figure_root(model_name=DATASET_NAME)
        / (
            f"{DATASET_NAME}_{value_name}{suffix}_"
            f"{species}_{year}_all_months.png"
        )
    )


def main() -> int:
    """Run Big Beautiful Dataset prediction map plots."""
    path = input_path(YEAR)
    print(f"Input: {path}")
    df = load_bbd(YEAR)
    agg_name = aggregation_name(AGG)
    plausibility_agg_name = aggregation_name(PLAUSIBILITY_AGG)

    for species in SPECIES:
        out_file = plot_prediction_df_map(
            df=df,
            value_col="species_use_log_pred",
            species=species,
            agg=AGG,
            title=f"{SPECIES_USE_STYLE.title} - {species} - {YEAR}",
            out_file=output_file(
                f"species_use_log_pred_{agg_name}",
                species,
                YEAR,
            ),
            style=SPECIES_USE_STYLE,
        )
        print(f"Saved: {out_file}")

        out_file = plot_prediction_df_map(
            df=df,
            value_col="risk_log_pred",
            species=species,
            agg=AGG,
            title=f"{REALIZED_RISK_STYLE.title} - {species} - {YEAR}",
            out_file=output_file(
                f"risk_log_pred_{agg_name}",
                species,
                YEAR,
            ),
            style=REALIZED_RISK_STYLE,
        )
        print(f"Saved: {out_file}")

        out_file = plot_hazard_df_map(
            df=df,
            species=species,
            agg=AGG,
            minimum_effort_unit=MINIMUM_EFFORT_UNIT,
            title=f"{HAZARD_STYLE.title} - {species} - {YEAR}",
            out_file=output_file(
                "hazard_log_pred",
                species,
                YEAR,
                agg_name,
            ),
            style=HAZARD_STYLE,
        )
        print(f"Saved: {out_file}")

        out_file = plot_hazard_plausibility_df_map(
            predictions=df,
            species=species,
            agg=AGG,
            plausibility_agg=PLAUSIBILITY_AGG,
            confidence_threshold=PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD,
            minimum_effort_unit=MINIMUM_EFFORT_UNIT,
            title=f"{HAZARD_PLAUSIBILITY_STYLE.title} - {species} - {YEAR}",
            out_file=output_file(
                "hazard_log_pred",
                species,
                YEAR,
                f"plausibility_threshold_{agg_name}_{plausibility_agg_name}",
            ),
            style=HAZARD_PLAUSIBILITY_STYLE,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
