"""Plot one gear-aware realized-risk example for BBAL set longlines."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd

from plot_prediction_maps import (
    AGG,
    REALIZED_RISK_STYLE,
    shared_binned_style,
)
from riskscape.config import paths
from riskscape.visualization.maps import aggregation_name, plot_prediction_df_map


YEAR = 2022
SPECIES = "BBAL"
GEAR_TYPE = "SET_LONGLINES"
MODEL_NAME = (
    "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_"
    "bayesian_gmm_k30"
)
PRODUCT_NAME = "joint"
EXPOSURE_PATH = (
    paths["data"]
    / "plot_exports"
    / "fishing_activity"
    / f"fishing_effort_by_gear_flag_{YEAR}.parquet"
)
OUTPUT_ROOT = paths["plots"] / "fishing_activity"


def prediction_path() -> Path:
    """Return the selected prediction partition."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / MODEL_NAME
        / PRODUCT_NAME
        / f"year={YEAR}"
        / "part.parquet"
    )


def load_gear_risk() -> pd.DataFrame:
    """Return BBAL risk joined to set-longline exposure."""
    pred_path = prediction_path()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction partition not found: {pred_path}")
    if not EXPOSURE_PATH.exists():
        raise FileNotFoundError(f"Gear/flag exposure table not found: {EXPOSURE_PATH}")

    query = """
        WITH exposure AS (
            SELECT
                CAST(h3 AS UBIGINT) AS h3,
                CAST(date AS DATE) AS date,
                SUM(fishing_hours)::DOUBLE AS fishing_hours,
                SUM(vessel_count)::DOUBLE AS vessel_count
            FROM read_parquet($exposure_path)
            WHERE gear_type = $gear_type
            GROUP BY h3, date
        ),
        risk AS (
            SELECT
                CAST(p.h3 AS UBIGINT) AS h3,
                CAST(p.date AS DATE) AS date,
                p.species,
                p.species_use_log_pred,
                CAST(e.fishing_hours * e.vessel_count AS FLOAT) AS fishing_activity,
                CAST(
                    p.species_use_log_pred
                    + ln(1.0 + e.fishing_hours * e.vessel_count)
                    AS FLOAT
                ) AS risk_log_pred
            FROM read_parquet($prediction_path) p
            INNER JOIN exposure e
            ON CAST(p.h3 AS UBIGINT) = e.h3
            AND CAST(p.date AS DATE) = e.date
            WHERE p.species = $species
        )
        SELECT *
        FROM risk
        ORDER BY date, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            {
                "exposure_path": str(EXPOSURE_PATH),
                "prediction_path": str(pred_path),
                "gear_type": GEAR_TYPE,
                "species": SPECIES,
            },
        ).df()


def output_path() -> Path:
    """Return output figure path."""
    agg_name = aggregation_name(AGG)
    gear_label = GEAR_TYPE.lower()
    return (
        OUTPUT_ROOT
        / f"{SPECIES.lower()}_{gear_label}_realized_risk_log_pred_{agg_name}_{YEAR}.png"
    )


def main() -> int:
    """Run the example map."""
    df = load_gear_risk()
    if df.empty:
        raise ValueError(f"No joined {SPECIES} {GEAR_TYPE} risk rows found")

    values = df.loc[df["risk_log_pred"] > 0, "risk_log_pred"]
    style = shared_binned_style(
        replace(
            REALIZED_RISK_STYLE,
            title="BBAL Set-Longline Realized Risk",
            colorbar_title="Realized Risk",
        ),
        values,
    )
    out_file = output_path()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    saved = plot_prediction_df_map(
        df=df,
        value_col="risk_log_pred",
        species=SPECIES,
        agg=AGG,
        title=f"BBAL realized risk from set longlines - {YEAR}",
        out_file=out_file,
        style=style,
    )
    print("Saved:", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
