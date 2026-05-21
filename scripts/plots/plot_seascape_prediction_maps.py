"""Plot seascape-conditioned species-use and risk maps."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import calendar
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import duckdb
import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import pairwise_distances

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.model.predict import output_columns
from riskscape.utils.dates import normalize_date_column
from riskscape.visualization.base_map import (
    MAP_CRS,
    MapBounds,
    OCEAN_COLOR,
    draw_reference_layers,
    load_reference_layers,
)
from riskscape.visualization.maps import (
    MINIMUM_EFFORT_UNIT,
    SPECIES_USE_LOG_COLOR_MAX,
    MapStyle,
    aggregation_name,
    figure_root,
    load_predictions,
    plot_prediction_df_map,
    summarize_h3,
)


YEAR = 2022
MODEL_NAME = "som_15x15_hierarchical_k30"
PREDICTION_MODEL = (
    "hybrid_presence_gate_extra_trees_som_hierarchical_k30_5fold_blockcv_bayesian_gmm_k30"
)
PREDICTION_PRODUCT = "joint"
SPECIES = ["BBAL", "SAFS"]
AGG = "non_zero_mean"
SEASCAPE_PREDICTION_PREFIX = "seascape"
BATCH_ROWS = 250_000
SOFT_TEMPERATURE_SCALE = 0.35

SURFACE_ROOT = paths["data"] / "modeling" / "seascape_species_use"
FEATURE_GRID_ROOT = paths["data"] / "modeling" / "feature_grid"
SEASCAPE_ROOT = paths["data"] / "modeling" / "environmental_regimes"
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
    color_max=None,
    color_quantile=1.0,
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
    """Return compact environmental-regime assignment path for one year."""
    seascape_file = (
        paths["data"]
        / "modeling"
        / "seascapes"
        / model_name
        / f"year={year}"
        / "part.parquet"
    )
    if seascape_file.exists():
        return seascape_file

    return SEASCAPE_ROOT / f"year={year}" / "part.parquet"


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


def prediction_figure_path(
    year: int,
    model_name: str,
    value_col: str,
    species: str,
    agg: str,
    month: int | None = None,
) -> Path:
    """Return the standard prediction-map output path."""
    month_label = f"month_{month:02d}" if month is not None else "all_months"
    return (
        figure_root(
            model_name=seascape_prediction_model_name(model_name),
            product_name=PREDICTION_PRODUCT,
        )
        / (
            f"{seascape_prediction_model_name(model_name)}_"
            f"{PREDICTION_PRODUCT}_{value_col}_{aggregation_name(agg)}_"
            f"{species}_{year}_{month_label}.png"
        )
    )


def monthly_matrix_figure_path(
    year: int,
    model_name: str,
    value_col: str,
    species: str,
    agg: str,
    scale_label: str = "absolute",
) -> Path:
    """Return the monthly-matrix output path."""
    return (
        figure_root(
            model_name=seascape_prediction_model_name(model_name),
            product_name=PREDICTION_PRODUCT,
        )
        / (
            f"{seascape_prediction_model_name(model_name)}_"
            f"{PREDICTION_PRODUCT}_{value_col}_{aggregation_name(agg)}_"
            f"{scale_label}_{species}_{year}_monthly_matrix.png"
        )
    )


def load_seascape_model(model_name: str):
    """Load a KMeans seascape model saved by the classifier script."""
    import classify_environmental_seascapes as seascapes

    sys.modules["__main__"].SeascapeModel = seascapes.SeascapeModel
    return joblib.load(seascape_model_path(model_name))


def seascape_assignment_column(model_name: str) -> str:
    """Return class column name for one seascape assignment table."""
    return "kmeans_k15" if model_name == "kmeans_k15" else "seascape"


def seascape_species_values(model_name: str, year: int) -> dict[str, np.ndarray]:
    """Return species-specific seascape-use vectors in cluster-id order."""
    summary_file = seascape_summary_path(model_name, year)
    if not summary_file.exists():
        raise FileNotFoundError(f"Seascape summary not found: {summary_file}")

    summary = pd.read_csv(summary_file)
    required = {
        "species",
        "seascape",
        "mean_log_residence_index",
    }
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"Seascape summary missing columns: {sorted(missing)}")

    values: dict[str, np.ndarray] = {}
    n_classes = int(summary["seascape"].max()) + 1
    for species, group in summary.groupby("species", sort=False):
        ordered = (
            group
            .sort_values("seascape")
            .set_index("seascape")
            .reindex(range(n_classes))
        )
        values[str(species)] = ordered["mean_log_residence_index"].fillna(
            0.0
        ).to_numpy(dtype="float64")

    return values


def build_hard_prediction_product(
    year: int,
    model_name: str,
) -> Path:
    """Write daily predictions by direct hard seascape assignment."""
    assignment_file = seascape_assignment_path(year, model_name)
    out_file = seascape_prediction_path_for_model(year, model_name)

    if not assignment_file.exists():
        raise FileNotFoundError(f"Seascape assignments not found: {assignment_file}")

    species_values = seascape_species_values(model_name, year)
    fishing = load_fishing_log(year)
    frames = []

    class_col = seascape_assignment_column(model_name)
    assignments = pd.read_parquet(
        assignment_file,
        columns=["h3", "date", class_col],
    )
    assignments = normalize_date_column(assignments)
    assignments["seascape"] = assignments[class_col].astype("int16")
    assignments = assignments.merge(fishing, on=["h3", "date"], how="left")
    assignments["fishing_activity_log"] = (
        assignments["fishing_activity_log"].fillna(0.0).astype("float32")
    )

    seascape_ids = assignments["seascape"].to_numpy(dtype="int16")
    fishing_raw = np.expm1(
        assignments["fishing_activity_log"].to_numpy(dtype="float64")
    )
    base = assignments[["h3", "date", "fishing_activity_log"]].reset_index(
        drop=True
    )

    for species, values in species_values.items():
        species_frame = base.copy()
        species_frame["species"] = species
        use_log = values[seascape_ids]
        species_frame["species_use_log_pred"] = use_log.astype("float32")
        species_frame["risk_log_pred"] = np.log1p(
            np.expm1(use_log) * fishing_raw
        ).astype("float32")
        frames.append(species_frame)

    predictions = normalize_date_column(pd.concat(frames, ignore_index=True))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    predictions[output_columns(seascape_prediction_model_name(model_name))].to_parquet(
        out_file,
        index=False,
        compression="zstd",
    )

    return out_file


def soft_membership_temperature(year: int, model_name: str) -> float:
    """Use the median nearest-centroid distance as the soft-membership scale."""
    path = seascape_assignment_path(year, model_name)
    if not path.exists():
        raise FileNotFoundError(f"Seascape assignments not found: {path}")

    with duckdb.connect(database=":memory:") as con:
        value = con.execute(
            "SELECT median(kmeans_k15_distance) FROM read_parquet($path)",
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
    if model_name == "kmeans_k15" or model_name.startswith("som_"):
        return build_hard_prediction_product(year=year, model_name=model_name)

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


def monthly_prediction_summary(
    year: int,
    model_name: str,
    species: str,
    value_col: str,
    agg: str,
) -> pd.DataFrame:
    """Return monthly H3 summaries from the seascape prediction product."""
    agg_name = aggregation_name(agg)
    product_path = seascape_prediction_path_for_model(year, model_name)
    if not product_path.exists():
        raise FileNotFoundError(f"Prediction product not found: {product_path}")

    if agg_name == "non_zero_mean":
        expr = f"avg({value_col}) FILTER (WHERE {value_col} > 0)"
    elif agg_name == "non_zero_median":
        expr = f"median({value_col}) FILTER (WHERE {value_col} > 0)"
    elif agg_name == "mean":
        expr = f"avg({value_col})"
    elif agg_name == "median":
        expr = f"median({value_col})"
    elif agg_name == "max":
        expr = f"max({value_col})"
    else:
        raise ValueError(f"Unsupported monthly matrix aggregation: {agg}")

    value_name = f"{value_col}_{agg_name}"
    query = f"""
        SELECT
            CAST(h3 AS UBIGINT) AS h3,
            CAST(month(date) AS INTEGER) AS month,
            {expr} AS {value_name}
        FROM read_parquet($product_path)
        WHERE species = $species
        GROUP BY h3, month
        ORDER BY month, h3
    """

    with duckdb.connect(database=":memory:") as con:
        return con.execute(
            query,
            {
                "product_path": str(product_path),
                "species": species,
            },
        ).fetchdf()


def matrix_norm(
    monthly: pd.DataFrame,
    value_name: str,
    style: MapStyle,
    power_gamma: float,
) -> colors.Normalize:
    """Return a shared color scale for one monthly matrix."""
    values = cast(pd.Series, monthly[value_name]).dropna()
    values = values[values > 0]
    if values.empty:
        raise ValueError(f"No positive monthly values found for {value_name}")

    if style.color_max is not None:
        vmax = float(style.color_max)
    else:
        vmax = float(values.quantile(style.color_quantile or 0.99))
    if vmax <= 0.0:
        vmax = float(values.max())
    if vmax <= 0.0:
        vmax = 1.0

    if power_gamma <= 0:
        raise ValueError("--power-gamma must be positive")
    if power_gamma != 1.0:
        return colors.PowerNorm(gamma=power_gamma, vmin=0.0, vmax=vmax)

    return colors.Normalize(vmin=0.0, vmax=vmax)


def apply_relative_scale(
    monthly: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    """Scale monthly values to the species-specific maximum."""
    out = monthly.copy()
    positive = cast(pd.Series, out[value_name]).dropna()
    max_value = float(positive.max()) if not positive.empty else 0.0
    if max_value <= 0.0:
        return out

    out[value_name] = out[value_name] / max_value
    return out


def matrix_scale_label(relative_scale: bool, power_gamma: float) -> str:
    """Return display-safe matrix scale text for output filenames."""
    scale = "relative" if relative_scale else "absolute"
    if power_gamma != 1.0:
        gamma_text = f"{power_gamma:g}".replace(".", "p")
        scale = f"{scale}_power{gamma_text}"
    return scale


def month_panel_title(month: int) -> str:
    """Return compact month title."""
    return calendar.month_abbr[month]


def plot_monthly_matrix_panel(
    ax: Axes,
    grid: gpd.GeoDataFrame,
    monthly: pd.DataFrame,
    value_name: str,
    month: int,
    norm: colors.Normalize,
    style: MapStyle,
    bounds: MapBounds,
    land: gpd.GeoDataFrame,
    coast: gpd.GeoDataFrame,
) -> None:
    """Draw one monthly matrix panel."""
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=0.35)

    month_mask = cast(pd.Series, monthly["month"]).eq(month)
    month_values = monthly.loc[month_mask].drop(columns=["month"])
    plot_gdf = grid.merge(month_values, on="h3", how="left")
    plot_gdf = plot_gdf.dropna(subset=[value_name])
    plot_gdf = plot_gdf[plot_gdf[value_name] > 0].copy()

    if not plot_gdf.empty:
        plot_gdf.plot(
            ax=ax,
            column=value_name,
            cmap=style.cmap,
            norm=norm,
            legend=False,
            edgecolor="none",
            linewidth=0,
        )

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[bounds.geometry()],
        crs=grid.crs or MAP_CRS,
    )
    draw_reference_layers(ax, bbox_gdf, land, coast)
    ax.set_title(month_panel_title(month), fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_monthly_matrix(
    monthly: pd.DataFrame,
    year: int,
    model_name: str,
    value_col: str,
    species: str,
    agg: str,
    style: MapStyle,
    relative_scale: bool,
    power_gamma: float,
) -> Path:
    """Plot a 12-panel monthly matrix for one species/value."""
    value_name = f"{value_col}_{aggregation_name(agg)}"
    if relative_scale:
        monthly = apply_relative_scale(monthly, value_name)
    grid = load_grid(uint64=True)
    land, coast = load_reference_layers()
    bounds = MapBounds.from_config()
    norm = matrix_norm(
        monthly=monthly,
        value_name=value_name,
        style=replace(style, color_max=1.0) if relative_scale else style,
        power_gamma=power_gamma,
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(10.5, 16),
        constrained_layout=False,
    )
    axes_flat = cast(list[Axes], axes.ravel().tolist())

    for month, ax in enumerate(axes_flat, start=1):
        plot_monthly_matrix_panel(
            ax=ax,
            grid=grid,
            monthly=monthly,
            value_name=value_name,
            month=month,
            norm=norm,
            style=style,
            bounds=bounds,
            land=land,
            coast=coast,
        )

    metric_label = (
        "Species Use" if value_col == "species_use_log_pred" else "Realized Risk"
    )
    scale_text = "Relative " if relative_scale else ""
    fig.suptitle(
        f"Monthly Seascape-Conditioned {scale_text}{metric_label} - "
        f"{species}, {year}",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.84,
        top=0.95,
        bottom=0.035,
        wspace=0.15,
        hspace=0.15,
    )

    cax = fig.add_axes((0.88, 0.20, 0.025, 0.60))
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(style.cmap)),
        cax=cax,
    )
    cbar.set_label(
        f"Relative {style.colorbar_title or metric_label}"
        if relative_scale
        else style.colorbar_title or metric_label
    )
    cbar.set_ticks([])
    cbar.ax.tick_params(which="both", length=0, labelleft=False, labelright=False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    out_file = monthly_matrix_figure_path(
        year=year,
        model_name=model_name,
        value_col=value_col,
        species=species,
        agg=agg,
        scale_label=matrix_scale_label(relative_scale, power_gamma),
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse an existing seascape prediction product instead of rebuilding it.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=None,
        help="Optional month numbers to plot instead of only the yearly map.",
    )
    parser.add_argument(
        "--monthly-matrix",
        action="store_true",
        help="Plot one 12-panel monthly matrix per species.",
    )
    parser.add_argument(
        "--matrix-values",
        nargs="+",
        default=["species_use_log_pred"],
        choices=("species_use_log_pred", "risk_log_pred"),
        help="Values to include in monthly matrix output.",
    )
    parser.add_argument(
        "--relative-scale",
        action="store_true",
        help="Scale each species monthly matrix by its own maximum value.",
    )
    parser.add_argument(
        "--power-gamma",
        type=float,
        default=1.0,
        help="PowerNorm gamma for monthly matrices; values below 1 stretch low values.",
    )

    return parser.parse_args()


def main() -> int:
    """Run seascape-conditioned prediction maps."""
    args = parse_args()
    out_product = seascape_prediction_path_for_model(args.year, args.model_name)
    if args.skip_build:
        if not out_product.exists():
            raise FileNotFoundError(f"Prediction product not found: {out_product}")
        print(f"Using existing standard prediction product: {out_product}")
    else:
        out_product = build_seascape_prediction_product(
            year=args.year,
            model_name=args.model_name,
            batch_rows=args.batch_rows,
        )
        print(f"Saved standard prediction product: {out_product}")

    if args.monthly_matrix:
        for species in args.species:
            for value_col in args.matrix_values:
                style = (
                    SPECIES_USE_STYLE
                    if value_col == "species_use_log_pred"
                    else REALIZED_RISK_STYLE
                )
                monthly = monthly_prediction_summary(
                    year=args.year,
                    model_name=args.model_name,
                    species=species,
                    value_col=value_col,
                    agg=args.agg,
                )
                out_file = plot_monthly_matrix(
                    monthly=monthly,
                    year=args.year,
                    model_name=args.model_name,
                    value_col=value_col,
                    species=species,
                    agg=args.agg,
                    style=style,
                    relative_scale=args.relative_scale,
                    power_gamma=args.power_gamma,
                )
                print(f"Saved: {out_file}")

        return 0

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

    months = args.months if args.months is not None else [None]

    for month in months:
        for species in args.species:
            month_text = (
                f", month {month:02d}" if month is not None else ""
            )
            out_file = plot_prediction_df_map(
                df=predictions,
                value_col="species_use_log_pred",
                species=species,
                month=month,
                agg=args.agg,
                title=(
                    f"Seascape-Conditioned Species Use — "
                    f"{species}, {args.year}{month_text}"
                ),
                out_file=prediction_figure_path(
                    year=args.year,
                    model_name=args.model_name,
                    value_col="species_use_log_pred",
                    species=species,
                    agg=args.agg,
                    month=month,
                ),
                style=species_style,
            )
            print(f"Saved: {out_file}")

            out_file = plot_prediction_df_map(
                df=predictions,
                value_col="risk_log_pred",
                species=species,
                month=month,
                agg=args.agg,
                title=(
                    f"Seascape-Conditioned Realized Risk — "
                    f"{species}, {args.year}{month_text}"
                ),
                out_file=prediction_figure_path(
                    year=args.year,
                    model_name=args.model_name,
                    value_col="risk_log_pred",
                    species=species,
                    agg=args.agg,
                    month=month,
                ),
                style=risk_style,
            )
            print(f"Saved: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
