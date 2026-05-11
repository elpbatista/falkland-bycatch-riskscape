"""Plot prediction maps from model prediction and plausibility outputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_COLOR_MAX,
    MapStyle,
    aggregation_name,
    figure_root,
    load_predictions,
    plot_hazard_df_map,
    plot_hazard_plausibility_df_map,
    plot_prediction_df_map,
    summarize_h3,
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
    title="Latent Risk",
    colorbar_title="Latent Risk",
)
HAZARD_PLAUSIBILITY_STYLE = replace(
    RISK_STYLE,
    title="Latent Plausible Risk",
    colorbar_title="Latent Plausible Risk",
)

SPECIES_USE_STYLE = MapStyle(
    title="Species Use",
    colorbar_title="Species Use",
    show_reference_map=False,
    min_display_value=0.0,
    color_min=0.0,
    color_max=SPECIES_USE_LOG_COLOR_MAX,
    color_quantile=None,
)


def aggregated_values(
    df: pd.DataFrame,
    value_col: str,
    species: str,
    agg: str,
) -> pd.Series:
    """Return map-level H3 values for one species/value."""
    summary = summarize_h3(
        df=df,
        value_col=value_col,
        species=species,
        agg=agg,
    )
    value_name = f"{value_col}_{aggregation_name(agg)}"
    return summary[value_name].dropna()


def shared_linear_style(
    style: MapStyle,
    values: pd.Series,
) -> MapStyle:
    """Return a fixed-scale continuous map style shared across panels."""
    if style.color_max is not None:
        return replace(style, color_quantile=None)

    positive = values[values > 0].dropna()
    if positive.empty:
        return style

    color_max = float(positive.quantile(0.99))
    if color_max <= 0:
        color_max = float(positive.max())

    return replace(
        style,
        color_min=0.0,
        color_max=color_max,
        color_quantile=None,
    )


def shared_binned_style(
    style: MapStyle,
    values: pd.Series,
) -> MapStyle:
    """Return a fixed-boundary binned style shared across panels."""
    if style.colorbar_labels is None:
        return style

    lower = style.color_min if style.color_min is not None else 0.0
    positive = values[values > lower].dropna()
    if positive.empty:
        return style

    quantiles = style.colorbar_quantiles or (0.0, 0.50, 0.90, 0.98, 1.0)
    bins = positive.quantile(quantiles).to_numpy(dtype="float64")
    bins[0] = lower

    if (bins[1:] <= bins[:-1]).any():
        upper = float(positive.quantile(0.99))
        if upper <= lower:
            upper = float(positive.max())
        if upper <= lower:
            upper = lower * 1.01 if lower > 0 else 1.0
        if style.color_scale == "log" and lower > 0:
            bins = np.geomspace(lower, upper, len(style.colorbar_labels) + 1)
        else:
            bins = np.linspace(lower, upper, len(style.colorbar_labels) + 1)

    return replace(
        style,
        colorbar_quantiles=None,
        colorbar_boundaries=tuple(float(value) for value in bins),
        color_max=float(bins[-1]),
    )


def shared_styles(
    predictions: pd.DataFrame,
    species: list[str],
    agg: str,
) -> tuple[MapStyle, MapStyle, MapStyle, MapStyle]:
    """Return fixed map styles for all maps produced in this run."""
    species_values = pd.concat(
        [
            aggregated_values(predictions, "species_use_log_pred", item, agg)
            for item in species
        ],
        ignore_index=True,
    )
    risk_values = pd.concat(
        [
            aggregated_values(predictions, "risk_log_pred", item, agg)
            for item in species
        ],
        ignore_index=True,
    )

    species_style = shared_linear_style(SPECIES_USE_STYLE, species_values)
    realized_style = shared_binned_style(REALIZED_RISK_STYLE, risk_values)
    hazard_style = shared_binned_style(HAZARD_STYLE, risk_values)
    plausible_hazard_style = shared_binned_style(
        HAZARD_PLAUSIBILITY_STYLE,
        risk_values,
    )

    return species_style, realized_style, hazard_style, plausible_hazard_style


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


def plausibility_path(
    year: int,
    product_name: str,
):
    """Return the available plausibility partition for one product/year."""
    root = paths["data"] / "modeling" / "plausibility"
    matches = sorted(root.glob(f"*/{product_name}/year={year}/part.parquet"))

    if not matches:
        raise FileNotFoundError(
            f"Plausibility file not found under: {root}"
        )

    if len(matches) > 1:
        raise ValueError(
            "Multiple plausibility files found: "
            + ", ".join(str(path) for path in matches)
        )

    return matches[0]


def load_plausibility_product(
    year: int,
    product_name: str,
) -> pd.DataFrame:
    """Load plausibility output for one product/year."""
    return pd.read_parquet(
        plausibility_path(year=year, product_name=product_name),
        columns=["h3", "species", "plausibility"],
    )


def output_file(
    model_name: str,
    product_name: str,
    value_name: str,
    species: str,
    year: int,
    suffix: str = "",
):
    """Return an output map path."""
    suffix = f"_{suffix}" if suffix else ""

    return (
        figure_root(model_name=model_name, product_name=product_name)
        / (
            f"{model_name}_{product_name}_{value_name}{suffix}_"
            f"{species}_{year}_all_months.png"
        )
    )


def map_title(base_title: str | None, species: str, year: int) -> str:
    """Return the display title for a species-year prediction map."""
    title = base_title or "Prediction"
    return f"{title} — {species}, {year}"


def main() -> int:
    """Run prediction map plots."""
    agg_name = aggregation_name(AGG)
    plausibility_agg_name = aggregation_name(PLAUSIBILITY_AGG)

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
            predictions = load_predictions(
                year=YEAR,
                model_name=model_name,
                product_name=product_name,
            )
            species_style, realized_style, hazard_style, plausible_hazard_style = (
                shared_styles(
                    predictions=predictions,
                    species=[item[1] for item in PREDICTION_PRODUCTS],
                    agg=AGG,
                )
            )
            plausibility = load_plausibility_product(
                year=YEAR,
                product_name=product_name,
            )

            out_file = plot_prediction_df_map(
                df=predictions,
                value_col="species_use_log_pred",
                species=species,
                agg=AGG,
                title=map_title(SPECIES_USE_STYLE.title, species, YEAR),
                out_file=output_file(
                    model_name,
                    product_name,
                    f"species_use_log_pred_{agg_name}",
                    species,
                    YEAR,
                ),
                style=species_style,
            )
            print(f"Saved: {out_file}")

            out_file = plot_prediction_df_map(
                df=predictions,
                value_col="risk_log_pred",
                species=species,
                agg=AGG,
                title=map_title(REALIZED_RISK_STYLE.title, species, YEAR),
                out_file=output_file(
                    model_name,
                    product_name,
                    f"risk_log_pred_{agg_name}",
                    species,
                    YEAR,
                ),
                style=realized_style,
            )
            print(f"Saved: {out_file}")

            out_file = plot_hazard_df_map(
                df=predictions,
                species=species,
                agg=AGG,
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                title=map_title(HAZARD_STYLE.title, species, YEAR),
                out_file=output_file(
                    model_name,
                    product_name,
                    "hazard_log_pred",
                    species,
                    YEAR,
                    agg_name,
                ),
                style=hazard_style,
            )
            print(f"Saved: {out_file}")

            out_file = plot_hazard_plausibility_df_map(
                predictions=predictions,
                plausibility_df=plausibility,
                species=species,
                agg=AGG,
                plausibility_agg=PLAUSIBILITY_AGG,
                confidence_threshold=PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD,
                minimum_effort_unit=MINIMUM_EFFORT_UNIT,
                title=map_title(
                    HAZARD_PLAUSIBILITY_STYLE.title,
                    species,
                    YEAR,
                ),
                out_file=output_file(
                    model_name,
                    product_name,
                    "hazard_log_pred",
                    species,
                    YEAR,
                    f"plausibility_threshold_{agg_name}_{plausibility_agg_name}",
                ),
                style=plausible_hazard_style,
            )
            print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
