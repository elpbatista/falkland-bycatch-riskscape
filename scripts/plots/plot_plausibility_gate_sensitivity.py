#!/usr/bin/env python3
"""Plot latent-risk sensitivity to the plausibility-gate strength."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import replace
from pathlib import Path
from typing import cast

import duckdb
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_prediction_maps import (
    AGG as PREDICTION_MAP_AGG,
    HAZARD_PLAUSIBILITY_STYLE,
    LOG_MINIMUM_EFFORT_UNIT,
    PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD,
    YEAR as PREDICTION_MAP_YEAR,
    load_aggregated_predictions,
    prediction_path,
    shared_styles,
)
from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.visualization.base_map import MapBounds, load_reference_layers
from riskscape.visualization.weekly_maps import format_week_panel
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    MapStyle,
    color_norm,
    draw_h3_value_panel,
    draw_prediction_colorbar,
    summarize_hazard_h3,
    summarize_plausibility_h3,
)


YEAR = 2022
MODEL_NAME = (
    "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_"
    "bayesian_gmm_k30"
)
PRODUCT_NAME = "joint"
SPECIES = ("BBAL", "SAFS")
GATE_CUTS = (0.0, 0.1, 0.25, 0.5)
SMALL_MULTIPLES_COLORBAR_POSITION = (0.935, 0.43, 0.018, 0.14)

OUT_NAME = (
    f"{MODEL_NAME}_{PRODUCT_NAME}_latent_risk_plausibility_gate_"
    f"sensitivity_{YEAR}_small_multiples.png"
)


def prediction_file() -> Path:
    """Return the prediction parquet for the selected year."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / MODEL_NAME
        / PRODUCT_NAME
        / f"year={YEAR}"
        / "part.parquet"
    )


def output_root() -> Path:
    """Return the prediction-plot output directory."""
    out = paths["plots"] / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    return out


def aggregate_sensitivity_query(cut: float) -> str:
    """Return SQL for one annual H3-level sensitivity prediction table."""
    return f"""
        WITH gated AS (
            SELECT
                h3,
                species,
                ln(
                    1.0
                    + (
                        greatest(exp(species_use_log_pred) - 1.0, 0.0)
                        / (1.0 - 0.100000 * (1.0 - plausibility))
                    )
                    * (1.0 - {cut:.6f} * (1.0 - plausibility))
                ) AS species_use_log_pred,
                plausibility
            FROM read_parquet($path, hive_partitioning=false)
            WHERE species = $species
        ),
        summarized AS (
            SELECT
                h3,
                species,
                COALESCE(
                    avg(
                        CASE
                            WHEN species_use_log_pred > 0
                            THEN species_use_log_pred
                            ELSE NULL
                        END
                    ),
                    0.0
                )::FLOAT AS species_use_log_pred_non_zero_mean,
                COALESCE(
                    avg(
                        CASE
                            WHEN plausibility > 0
                            THEN plausibility
                            ELSE NULL
                        END
                    ),
                    0.0
                )::FLOAT AS plausibility_non_zero_mean
            FROM gated
            GROUP BY h3, species
        )
        SELECT
            h3,
            species,
            species_use_log_pred_non_zero_mean AS species_use_log_pred,
            plausibility_non_zero_mean AS plausibility
        FROM summarized
    """


def load_production_aggregated_predictions() -> pd.DataFrame:
    """Load the exact aggregated prediction table used by plot_prediction_maps."""
    path = prediction_path(
        PREDICTION_MAP_YEAR,
        MODEL_NAME,
        PRODUCT_NAME,
    )
    return load_aggregated_predictions(
        path=path,
        species_values=list(SPECIES),
        agg=PREDICTION_MAP_AGG,
        plausibility_agg=PREDICTION_MAP_AGG,
    )


def load_aggregated_sensitivity_predictions(cut: float) -> pd.DataFrame:
    """Load annual H3-level predictions for one plausibility-gate setting."""
    if np.isclose(cut, 0.1):
        return load_production_aggregated_predictions()

    path = prediction_file()
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction table: {path}")

    frames = []
    with duckdb.connect(database=":memory:") as con:
        con.execute("PRAGMA temp_directory='/private/tmp/duckdb'")
        con.execute("PRAGMA memory_limit='4GB'")
        for species in SPECIES:
            frame = con.execute(
                aggregate_sensitivity_query(cut),
                {"path": str(path), "species": species},
            ).fetchdf()
            frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out["h3"] = out["h3"].astype("uint64")
    return out


def build_surface(predictions: pd.DataFrame, cut: float) -> pd.DataFrame:
    """Build the same latent plausible-risk H3 surface used by production maps."""
    frames = []
    for species in SPECIES:
        hazard = summarize_hazard_h3(
            df=predictions,
            species=species,
            agg=PREDICTION_MAP_AGG,
            minimum_effort_unit=MINIMUM_EFFORT_UNIT,
        )
        plausibility = summarize_plausibility_h3(
            df=predictions,
            species=species,
            agg=PREDICTION_MAP_AGG,
            confidence_threshold=PLAUSIBLE_RISK_CONFIDENCE_THRESHOLD,
        )
        plausibility_col = f"plausibility_{PREDICTION_MAP_AGG}"
        surface = hazard.merge(plausibility, on="h3", how="left")
        surface["low_plausibility"] = (
            surface[plausibility_col].fillna(0.0) <= 0.0
        )
        surface["species"] = species
        surface["gate_cut"] = cut
        frames.append(
            surface[["h3", "species", "gate_cut", "hazard_log_pred", "low_plausibility"]]
        )

    return pd.concat(frames, ignore_index=True)


def load_sensitivity_surfaces() -> pd.DataFrame:
    """Load annual H3-level latent-risk summaries for all gate settings."""
    frames = []
    for cut in GATE_CUTS:
        predictions = load_aggregated_sensitivity_predictions(cut)
        frames.append(build_surface(predictions, cut))

    out = pd.concat(frames, ignore_index=True)
    out["h3"] = out["h3"].astype("uint64")
    return out


def sensitivity_style(values: pd.Series) -> MapStyle:
    """Return the shared latent-risk style for all panels."""
    production_predictions = load_production_aggregated_predictions()
    _, _, _, plausible_hazard_style = shared_styles(
        predictions=production_predictions,
        species=list(SPECIES),
        agg=PREDICTION_MAP_AGG,
    )
    return replace(
        plausible_hazard_style,
        title=None,
        colorbar_title=HAZARD_PLAUSIBILITY_STYLE.colorbar_title,
        min_display_value=LOG_MINIMUM_EFFORT_UNIT,
        color_min=LOG_MINIMUM_EFFORT_UNIT,
        show_reference_map=False,
    )


def risk_norm(style: MapStyle) -> colors.BoundaryNorm:
    """Return the binned latent-risk color scale."""
    if style.colorbar_boundaries is None:
        raise ValueError("Sensitivity style requires colorbar boundaries")
    norm = color_norm(
        pd.Series(style.colorbar_boundaries, dtype="float64"),
        style,
    )
    if not isinstance(norm, colors.BoundaryNorm):
        raise TypeError("Sensitivity style requires BoundaryNorm")
    return norm


def panel_title(species: str, cut: float) -> str:
    """Return panel title."""
    return f"{species} - c_s = {cut:g}"


def plot_sensitivity_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    surfaces: pd.DataFrame,
    species: str,
    cut: float,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one plausibility-gate sensitivity panel."""
    format_week_panel(
        ax,
        species=species,
        week=0,
        bounds=bounds,
        title=panel_title(species, cut),
    )

    subset = surfaces[
        (surfaces["species"] == species)
        & np.isclose(surfaces["gate_cut"], cut)
    ].copy()
    subset["low_plausibility"] = subset["low_plausibility"].fillna(False).astype(bool)
    draw_h3_value_panel(
        ax=ax,
        grid=grid,
        values=subset[["h3", "hazard_log_pred", "low_plausibility"]],
        value_col="hazard_log_pred",
        norm=norm,
        style=style,
        bounds=bounds,
        land=land,
        coast=coast,
        outline_col="low_plausibility",
        draw_bathymetry=True,
    )


def add_risk_colorbar(
    ax: Axes,
    norm: colors.BoundaryNorm,
    style: MapStyle,
    cax: Axes | None = None,
) -> None:
    """Add the same labeled latent-risk colorbar used by prediction maps."""
    draw_prediction_colorbar(
        ax=ax,
        value_col="hazard_log_pred",
        norm=norm,
        style=style,
        cax=cax,
    )


def output_path() -> Path:
    """Return output figure path."""
    return output_root() / OUT_NAME


def plot_small_multiples(surfaces: pd.DataFrame) -> Path:
    """Plot species by plausibility-gate sensitivity small multiples."""
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    style = sensitivity_style(surfaces["hazard_log_pred"])
    norm = risk_norm(style)

    fig, axes = plt.subplots(
        nrows=len(SPECIES),
        ncols=len(GATE_CUTS),
        figsize=(4.0 * len(GATE_CUTS), 5.1 * len(SPECIES)),
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)

    for row, species in enumerate(SPECIES):
        for col, cut in enumerate(GATE_CUTS):
            plot_sensitivity_panel(
                ax=cast(Axes, axes_array[row, col]),
                grid=grid,
                surfaces=surfaces,
                species=species,
                cut=cut,
                norm=norm,
                style=style,
                bounds=bounds,
                land=land,
                coast=coast,
            )

    fig.suptitle(
        f"Latent-Risk Sensitivity to Plausibility Gate - {YEAR}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.91,
        top=0.93,
        bottom=0.035,
        wspace=0.09,
        hspace=0.16,
    )

    cax = fig.add_axes(SMALL_MULTIPLES_COLORBAR_POSITION)
    add_risk_colorbar(
        ax=cast(Axes, axes_array[0, -1]),
        norm=norm,
        style=style,
        cax=cast(Axes, cax),
    )

    out_file = output_path()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def main() -> int:
    """Run the sensitivity plot workflow."""
    surfaces = load_sensitivity_surfaces()
    out_file = plot_small_multiples(surfaces)
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
