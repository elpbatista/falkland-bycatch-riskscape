"""Plot seascape-conditioned species-use and risk maps."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import pairwise_distances

from riskscape.config import paths
from riskscape.model.predict import output_columns
from riskscape.utils.dates import normalize_date_column
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_COLOR_MAX,
    MapStyle,
    aggregation_name,
    load_predictions,
    plot_prediction_map,
    summarize_h3,
)


YEAR = 2022
MODEL_NAME = "kmeans_k10"
PREDICTION_MODEL = (
    "hybrid_presence_gate_extra_trees_kmeans_k15_blockcv_bayesian_gmm_k30"
)
PREDICTION_PRODUCT = "joint"
SPECIES = ["BBAL", "SAFS"]
AGG = "non_zero_mean"
SEASCAPE_PREDICTION_PREFIX = "seascape"
BATCH_ROWS = 250_000
SOFT_TEMPERATURE_SCALE = 0.35

SURFACE_ROOT = paths["data"] / "modeling" / "seascape_species_use"
FEATURE_GRID_ROOT = paths["data"] / "modeling" / "feature_grid"
SEASCAPE_ROOT = paths["data"] / "modeling" / "seascapes"
SEASCAPE_MODEL_ROOT = paths["data"] / "modeling" / "models" / "seascapes"
SEASCAPE_SUMMARY_ROOT = paths["data"] / "plot_exports" / "seascapes"
FISHING_ROOT = paths["data"] / "modeling" / "fishing_training"
SOURCE_PREDICTION_ROOT = (
    paths["data"]
    / "modeling"
    / "predictions"
    / PREDICTION_MODEL
    / PREDICTION_PRODUCT
)
def seascape_prediction_model_name(model_name: str) -> str:
    """Return output prediction product name for one seascape model."""
    return f"{SEASCAPE_PREDICTION_PREFIX}_{model_name}"


REALIZED_RISK_STYLE = MapStyle(
    color_scale="log",
    title="Realized Risk",
    colorbar_title="Realized Risk",
    alpha_scale=False,
    alpha=0.90,
    show_reference_map=False,
    min_display_value=float(np.log1p(MINIMUM_EFFORT_UNIT)),
    color_min=float(np.log1p(MINIMUM_EFFORT_UNIT)),
    colorbar_labels=("Low", "Mod", "High", "Xtrm"),
    colorbar_quantiles=(0.0, 0.50, 0.90, 0.98, 1.0),
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


def shared_linear_style(style: MapStyle, values: pd.Series) -> MapStyle:
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


def shared_binned_style(style: MapStyle, values: pd.Series) -> MapStyle:
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

    if np.any(bins[1:] <= bins[:-1]):
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
) -> tuple[MapStyle, MapStyle]:
    """Return fixed species-use and risk styles for the plotted map set."""
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

    return (
        shared_linear_style(SPECIES_USE_STYLE, species_values),
        shared_binned_style(REALIZED_RISK_STYLE, risk_values),
    )


def surface_path(year: int, model_name: str) -> Path:
    """Return seascape-conditioned species-use surface path."""
    return SURFACE_ROOT / model_name / f"year={year}" / "part.parquet"


def feature_grid_path(year: int) -> Path:
    """Return feature-grid path for one year."""
    return FEATURE_GRID_ROOT / f"year={year}" / "part.parquet"


def seascape_assignment_path(year: int, model_name: str) -> Path:
    """Return hard seascape assignment path for one year."""
    return SEASCAPE_ROOT / model_name / f"year={year}" / "part.parquet"


def seascape_model_path(model_name: str) -> Path:
    """Return fitted seascape model path."""
    return SEASCAPE_MODEL_ROOT / f"{model_name}.joblib"


def seascape_summary_path(model_name: str, year: int) -> Path:
    """Return species-use summary by seascape for one map year."""
    year_file = (
        SEASCAPE_SUMMARY_ROOT
        / f"predicted_species_use_by_seascape_{model_name}_{year}.csv"
    )
    if year_file.exists():
        return year_file

    return (
        SEASCAPE_SUMMARY_ROOT
        / f"predicted_species_use_by_seascape_{model_name}_2014-2023.csv"
    )


def fishing_path(year: int) -> Path:
    """Return fishing training path for one year."""
    return FISHING_ROOT / f"year={year}" / "part.parquet"


def prediction_path(year: int) -> Path:
    """Return original prediction partition path for fishing exposure."""
    return SOURCE_PREDICTION_ROOT / f"year={year}" / "part.parquet"


def seascape_prediction_path_for_model(year: int, model_name: str) -> Path:
    """Return standard-schema seascape prediction partition path."""
    return (
        paths["data"]
        / "modeling"
        / "predictions"
        / seascape_prediction_model_name(model_name)
        / PREDICTION_PRODUCT
        / f"year={year}"
        / "part.parquet"
    )


def load_seascape_model(model_name: str):
    """Load a KMeans seascape model saved by the classifier script."""
    import classify_environmental_seascapes as seascapes

    sys.modules["__main__"].SeascapeModel = seascapes.SeascapeModel
    return joblib.load(seascape_model_path(model_name))


def seascape_species_values(model_name: str, year: int) -> dict[str, np.ndarray]:
    """Return species-specific seascape-use vectors in cluster-id order."""
    summary_file = seascape_summary_path(model_name, year)
    if not summary_file.exists():
        raise FileNotFoundError(f"Seascape summary not found: {summary_file}")

    summary = pd.read_csv(summary_file)
    values: dict[str, np.ndarray] = {}
    for species, group in summary.groupby("species", sort=False):
        ordered = group.sort_values("seascape")
        values[str(species)] = ordered["median_log_residence_index"].to_numpy(
            dtype="float64"
        )

    return values


def soft_membership_temperature(year: int, model_name: str) -> float:
    """Use the median nearest-centroid distance as the soft-membership scale."""
    path = seascape_assignment_path(year, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Seascape assignments not found: {path}")

    with duckdb.connect(database=":memory:") as con:
        value = con.execute(
            "SELECT median(seascape_distance) FROM read_parquet($path)",
            {"path": str(path)},
        ).fetchone()[0]

    if value is None or value <= 0:
        raise ValueError("Cannot derive positive soft-membership temperature")

    return float(value)


def load_fishing_log(year: int) -> pd.DataFrame:
    """Load H3/day fishing activity in the existing log convention."""
    path = fishing_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Fishing data not found: {path}")

    fishing = pd.read_parquet(path, columns=["h3", "date", "fishing_activity"])
    fishing["fishing_activity_log"] = np.log1p(
        fishing["fishing_activity"].clip(lower=0.0)
    ).astype("float32")
    return fishing[["h3", "date", "fishing_activity_log"]]


def build_soft_prediction_batch(
    batch: pd.DataFrame,
    payload,
    species_values: dict[str, np.ndarray],
    fishing: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    """Build seascape-conditioned predictions for one H3/day batch."""
    features = payload.features
    out = batch.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    if out.empty:
        return pd.DataFrame(
            columns=["h3", "date", "species", "species_use_log_pred"]
        )

    x = out[features].to_numpy(dtype="float64")
    x_scaled = payload.scaler.transform(x)
    distances = pairwise_distances(x_scaled, payload.model.cluster_centers_)
    scaled = -0.5 * np.square(distances / temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights = weights / weights.sum(axis=1, keepdims=True)

    base = out[["h3", "date"]].reset_index(drop=True)
    frames = []
    for species, values in species_values.items():
        species_frame = base.copy()
        species_frame["species"] = species
        species_frame["species_use_log_pred"] = (
            weights @ values
        ).astype("float32")
        frames.append(species_frame)

    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.merge(fishing, on=["h3", "date"], how="left")
    predictions["fishing_activity_log"] = (
        predictions["fishing_activity_log"].fillna(0.0).astype("float32")
    )
    species_use = np.expm1(
        predictions["species_use_log_pred"].to_numpy(dtype="float64")
    )
    fishing = np.expm1(
        predictions["fishing_activity_log"].to_numpy(dtype="float64")
    )
    predictions["risk_log_pred"] = np.log1p(species_use * fishing).astype("float32")
    return predictions


def combine_prediction_parts(parts_dir: Path, out_file: Path) -> None:
    """Combine batch parquet parts into the standard prediction partition file."""
    with duckdb.connect(database=":memory:") as con:
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{parts_dir.as_posix()}/*.parquet')
            )
            TO '{out_file.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )


def build_seascape_prediction_product(
    year: int,
    model_name: str,
    batch_rows: int,
) -> Path:
    """Write a standard prediction product for soft seascape-conditioned maps."""
    feature_file = feature_grid_path(year)
    out_file = seascape_prediction_path_for_model(year, model_name)

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature grid not found: {feature_file}")

    payload = load_seascape_model(model_name)
    species_values = seascape_species_values(model_name, year)
    fishing = load_fishing_log(year)
    temperature = (
        soft_membership_temperature(year, model_name)
        * SOFT_TEMPERATURE_SCALE
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out_file.parent / "_soft_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    parquet_file = pq.ParquetFile(feature_file)
    columns = ["h3", "date", *payload.features]
    part_index = 0
    for record_batch in parquet_file.iter_batches(
        batch_size=batch_rows,
        columns=columns,
    ):
        batch = record_batch.to_pandas()
        predictions = build_soft_prediction_batch(
            batch=batch,
            payload=payload,
            species_values=species_values,
            fishing=fishing,
            temperature=temperature,
        )
        if predictions.empty:
            continue
        out = normalize_date_column(
            predictions[output_columns(seascape_prediction_model_name(model_name))]
        )
        out.to_parquet(
            parts_dir / f"part_{part_index:05d}.parquet",
            index=False,
            compression="zstd",
        )
        part_index += 1

    if part_index == 0:
        raise ValueError("No seascape-conditioned prediction rows were written")

    combine_prediction_parts(parts_dir, out_file)
    shutil.rmtree(parts_dir)

    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot seascape-conditioned species-use and realized-risk maps.",
    )
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--species",
        nargs="+",
        default=SPECIES,
        help="Species codes to plot.",
    )
    parser.add_argument(
        "--agg",
        default=AGG,
        choices=("non_zero_median", "non_zero_mean", "mean", "median", "max"),
        help="Single H3/day vertical-stack aggregation to generate for this run.",
    )
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)

    return parser.parse_args()


def main() -> int:
    """Run seascape-conditioned prediction maps."""
    args = parse_args()
    out_product = build_seascape_prediction_product(
        year=args.year,
        model_name=args.model_name,
        batch_rows=args.batch_rows,
    )
    print(f"Saved standard prediction product: {out_product}")
    predictions = load_predictions(
        year=args.year,
        model_name=seascape_prediction_model_name(args.model_name),
        product_name=PREDICTION_PRODUCT,
    )
    species_style, risk_style = shared_styles(
        predictions=predictions,
        species=list(args.species),
        agg=args.agg,
    )

    for species in args.species:
        out_file = plot_prediction_map(
            year=args.year,
            model_name=seascape_prediction_model_name(args.model_name),
            product_name=PREDICTION_PRODUCT,
            value_col="species_use_log_pred",
            species=species,
            agg=args.agg,
            title=f"Seascape-Conditioned Species Use — {species}, {args.year}",
            style=species_style,
        )
        print(f"Saved: {out_file}")

        out_file = plot_prediction_map(
            year=args.year,
            model_name=seascape_prediction_model_name(args.model_name),
            product_name=PREDICTION_PRODUCT,
            value_col="risk_log_pred",
            species=species,
            agg=args.agg,
            title=f"Seascape-Conditioned Realized Risk — {species}, {args.year}",
            style=risk_style,
        )
        print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
