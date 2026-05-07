"""Generate model predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import (
    FEATURES,
    FISHING_TARGET,
    SPECIES_TARGET,
    available_years,
    load_feature_grid,
)
from riskscape.model.train import MODEL_NAMES


# Prediction mode options:
#   "single" -> run one model family selected by ACTIVE_MODEL_NAME
#   "all"    -> run every model family listed in MODEL_NAMES
#   "hybrid" -> combine one ML model and one Bayesian/GMM model
PREDICTION_MODE = "hybrid"

# Used only when PREDICTION_MODE = "single".
# ACTIVE_MODEL_NAME options (must match MODEL_NAMES from train.py):
#   "hist_gradient_boosting"
#   "random_forest"
#   "extra_trees"
#   "gmm"
#   "bayesian_gmm"
ACTIVE_MODEL_NAME = "extra_trees"

# Used only when PREDICTION_MODE = "hybrid".
HYBRID_ML_MODEL_NAME = "extra_trees"
HYBRID_BAYESIAN_MODEL_NAME = "bayesian_gmm"

# Hybrid strategy options:
#   "constant"      -> fixed weighted blend
#   "spatial_alpha" -> blend weight increases with historical cell support
#   "conditional"   -> ML where historical support exists, Bayesian elsewhere
#   "presence_gate" -> Bayesian/GMM likelihood gates ML intensity
HYBRID_MODE = "presence_gate"

HYBRID_ALPHA = 0.8
HYBRID_MIN_ALPHA = 0.1
HYBRID_MAX_ALPHA = 0.9
HYBRID_SUPPORT_K = 5.0
HYBRID_SUPPORT_THRESHOLD = 1.0
# Maximum fraction cut from the Extra Trees species-use prediction
# when Bayesian/GMM plausibility is zero.
HYBRID_GATE_MAX_CUTS = {
    "BBAL": 0.00,
    "SAFS": 0.01,
}

BATCH_ROWS = 250_000
# PREDICTION_PRODUCTS = ["joint", "bbal", "safs"]
PREDICTION_PRODUCTS = ["joint"]

MODEL_DIR = paths["data"] / "modeling" / "models"
PREDICTION_ROOT = paths["data"] / "modeling" / "predictions"


def load_model_payload(path: Path):
    """Load model payload."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)


def species_model_path(species: str, model_name: str) -> Path:
    """Return model path."""
    return (
        MODEL_DIR
        / model_name
        / f"species_model_{species.lower()}.joblib"
    )


def product_model_path(product_name: str, model_name: str) -> Path:
    """Return product model path."""
    return (
        MODEL_DIR
        / model_name
        / f"species_model_{product_name}.joblib"
    )


def load_product_payloads(model_name: str) -> dict[str, dict]:
    """Load joint, BBAL, and SAFS product models."""
    payloads = {}

    for product_name in PREDICTION_PRODUCTS:
        path = product_model_path(product_name, model_name)
        payload = load_model_payload(path)
        validate_model_features(payload)
        payloads[product_name] = payload

        payload_model_name = payload.get("model_name", "unknown")
        payload_model_type = payload.get("model_type", "unknown")
        print(
            f"Loaded {product_name} model: "
            f"{payload_model_name} ({payload_model_type})"
        )

    return payloads


def load_species_payloads(
    species_values: list[str],
    model_name: str,
) -> dict[str, dict]:
    """Load species models."""
    payloads = {}

    for species in species_values:
        path = species_model_path(species, model_name)
        payload = load_model_payload(path)
        payloads[species] = payload

        payload_model_name = payload.get("model_name", "unknown")
        print(f"Loaded {species} model: {payload_model_name}")

    return payloads


def hybrid_output_name() -> str:
    """Return hybrid output folder name."""
    return (
        f"hybrid_{HYBRID_MODE}_"
        f"{HYBRID_ML_MODEL_NAME}_{HYBRID_BAYESIAN_MODEL_NAME}"
    )


def species_training_paths() -> list[Path]:
    """Return species training partition paths."""
    root = paths["data"] / "modeling" / "species_training"
    return sorted(root.glob("year=*/part.parquet"))


def load_hybrid_support() -> dict[str, pd.DataFrame]:
    """Load historical positive-use support by species and H3 cell."""
    frames = []

    cols = ["h3", "species", SPECIES_TARGET]

    for path in species_training_paths():
        df = pd.read_parquet(path, columns=cols)
        df = df[df[SPECIES_TARGET] > 0]

        if not df.empty:
            frames.append(df)

    if not frames:
        return {}

    support = pd.concat(frames, ignore_index=True)
    support = (
        support
        .groupby(["species", "h3"], as_index=False)
        .agg(
            support_count=(SPECIES_TARGET, "size"),
            support_sum=(SPECIES_TARGET, "sum"),
        )
    )

    out = {}

    for species, group in support.groupby("species", sort=True):
        out[species] = group[["h3", "support_count", "support_sum"]].copy()

    return out


def prediction_model_name(model_name: str) -> str:
    """Return output model name."""
    if model_name != "hybrid":
        return model_name
    return hybrid_output_name()


def output_dir(model_name: str, year: int) -> Path:
    return PREDICTION_ROOT / prediction_model_name(model_name) / f"year={year}"


def product_output_dir(
    model_name: str,
    product_name: str,
    year: int,
) -> Path:
    return (
        PREDICTION_ROOT
        / prediction_model_name(model_name)
        / product_name
        / f"year={year}"
    )


def merged_output_path(model_name: str, year: int) -> Path:
    return output_dir(model_name, year) / "part.parquet"


def product_merged_output_path(
    model_name: str,
    product_name: str,
    year: int,
) -> Path:
    return product_output_dir(model_name, product_name, year) / "part.parquet"


def clean_output_dir(model_name: str, year: int) -> None:
    out_dir = output_dir(model_name, year)

    if not out_dir.exists():
        return

    for path in out_dir.glob("part-*.parquet"):
        path.unlink()

    merged = merged_output_path(model_name, year)
    if merged.exists():
        merged.unlink()


def clean_product_output_dir(
    model_name: str,
    product_name: str,
    year: int,
) -> None:
    out_dir = product_output_dir(model_name, product_name, year)

    if not out_dir.exists():
        return

    for path in out_dir.glob("part-*.parquet"):
        path.unlink()

    merged = product_merged_output_path(model_name, product_name, year)
    if merged.exists():
        merged.unlink()


def iter_batches(df: pd.DataFrame, batch_rows: int):
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def validate_model_features(species_payload) -> None:
    payload_features = species_payload.get("features", FEATURES)

    if list(payload_features) != list(FEATURES):
        raise ValueError(
            "Model features do not match current FEATURES. "
            f"Model: {list(payload_features)}. "
            f"Current: {list(FEATURES)}."
        )


def feature_matrix(batch: pd.DataFrame) -> pd.DataFrame:
    x = batch[FEATURES].replace([np.inf, -np.inf], np.nan)

    if x.isna().any().any():
        missing = x.columns[x.isna().any()].tolist()
        raise ValueError(
            "NaN values detected in prediction features. "
            "Rebuild feature_grid with neighbor filling. "
            f"Missing columns: {missing}"
        )

    return x

# NEW FUNCTION: Drop prediction rows with invalid feature values
def drop_invalid_prediction_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Drop prediction rows with invalid feature values."""
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    mask = out[FEATURES].isna().any(axis=1)

    if not mask.any():
        return out

    missing_counts = out.loc[mask, FEATURES].isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(
        ascending=False,
    )

    print(
        f"Dropping {int(mask.sum())} prediction rows for {year} "
        "because feature values are missing."
    )
    print(missing_counts)

    return out.loc[~mask].reset_index(drop=True)


def predict_species(
    batch: pd.DataFrame,
    species_payload,
) -> np.ndarray:
    validate_model_features(species_payload)

    model = species_payload["model"]
    pred = model.predict(feature_matrix(batch))
    pred = np.maximum(pred, 0.0)

    return pred.astype("float32")


def normalize_log_density(log_density: np.ndarray, model) -> np.ndarray:
    """Normalize model log-density to plausibility range 0-1."""
    denom = model.max_log_density - model.min_log_density

    if denom <= 0:
        return np.zeros_like(log_density, dtype="float32")

    plausibility = (log_density - model.min_log_density) / denom
    plausibility = np.clip(plausibility, 0.0, 1.0)

    return plausibility.astype("float32")


def predict_plausibility(
    batch: pd.DataFrame,
    payload: dict,
) -> np.ndarray:
    """Predict environmental plausibility from a density payload."""
    validate_model_features(payload)

    model = payload["model"]
    log_density = model.predict_log_density(feature_matrix(batch))

    return normalize_log_density(log_density, model)


def predict_hybrid_species(
    batch: pd.DataFrame,
    ml_payload,
    bayesian_payload,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ML and Bayesian species use."""
    ml_pred = predict_species(batch, ml_payload)
    bayesian_pred = predict_species(batch, bayesian_payload)

    return ml_pred, bayesian_pred


def add_support_columns(
    df: pd.DataFrame,
    support: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add historical support columns to prediction rows."""
    if support is None or support.empty:
        out = df.copy()
        out["support_count"] = 0.0
        out["support_sum"] = 0.0
        return out

    out = df.merge(
        support,
        on="h3",
        how="left",
    )

    out["support_count"] = out["support_count"].fillna(0.0).astype("float32")
    out["support_sum"] = out["support_sum"].fillna(0.0).astype("float32")

    return out


def hybrid_alpha(support_count: np.ndarray) -> np.ndarray:
    """Return alpha values for hybrid prediction."""
    support_count = support_count.astype("float32")

    if HYBRID_MODE == "constant":
        return np.full_like(
            support_count,
            HYBRID_ALPHA,
            dtype="float32",
        )

    if HYBRID_MODE == "spatial_alpha":
        alpha = support_count / (support_count + HYBRID_SUPPORT_K)
        alpha = np.clip(alpha, HYBRID_MIN_ALPHA, HYBRID_MAX_ALPHA)
        return alpha.astype("float32")

    if HYBRID_MODE == "conditional":
        alpha = np.where(
            support_count >= HYBRID_SUPPORT_THRESHOLD,
            1.0,
            0.0,
        )
        return alpha.astype("float32")

    raise ValueError(f"Unknown HYBRID_MODE: {HYBRID_MODE}")


def blend_hybrid_predictions(
    ml_pred: np.ndarray,
    bayesian_pred: np.ndarray,
    support_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend ML and Bayesian predictions using selected hybrid mode."""
    alpha = hybrid_alpha(support_count)
    pred = alpha * ml_pred + (1.0 - alpha) * bayesian_pred

    return alpha.astype("float32"), pred.astype("float32")


def presence_gate(
    plausibility: np.ndarray,
    species: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return species-specific effective plausibility gate."""
    species_values = pd.Series(species)
    max_cut = (
        species_values.map(HYBRID_GATE_MAX_CUTS)
        .fillna(0.0)
        .to_numpy(dtype="float32")
    )
    gate = 1.0 - max_cut * (1.0 - plausibility)

    return gate.astype("float32")


def apply_presence_gate(
    species_use_log_pred: np.ndarray,
    plausibility: np.ndarray,
    species: pd.Series | np.ndarray,
) -> np.ndarray:
    """Gate log-space species-use predictions by plausibility."""
    species_use = np.expm1(species_use_log_pred)
    species_use = np.maximum(species_use, 0.0)

    gate = presence_gate(plausibility, species)
    hybrid_species_use = species_use * gate
    hybrid_species_use_log = np.log1p(hybrid_species_use)

    return hybrid_species_use_log.astype("float32")


def build_species_predictions(
    batch: pd.DataFrame,
    species_payloads: dict[str, dict],
) -> pd.DataFrame:
    frames = []

    for species, payload in species_payloads.items():
        out = batch.copy()
        out["species"] = species
        out["species_use_log_pred"] = predict_species(out, payload)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def build_joint_predictions(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Predict all species from one joint model payload."""
    validate_model_features(payload)

    encoder = payload["encoder"]
    model = payload["model"]
    frames = []

    for species in encoder.categories_[0]:
        out = batch.copy()
        out["species"] = species

        x_species = encoder.transform(out[["species"]])
        x_features = feature_matrix(out).to_numpy()
        x = np.hstack([x_species, x_features])

        pred = model.predict(x)
        out["species_use_log_pred"] = np.maximum(pred, 0.0).astype("float32")
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def build_product_predictions(
    batch: pd.DataFrame,
    product_name: str,
    payload: dict,
) -> pd.DataFrame:
    """Predict one product payload."""
    if payload.get("model_type") == "joint_species":
        return build_joint_predictions(batch, payload)

    species = payload.get("species", product_name.upper())
    return build_species_predictions(batch, {species: payload})


def build_joint_plausibility(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Predict plausibility for all species from one joint density payload."""
    validate_model_features(payload)

    encoder = payload["encoder"]
    model = payload["model"]
    frames = []

    for species in encoder.categories_[0]:
        out = batch.copy()
        out["species"] = species

        x_species = encoder.transform(out[["species"]])
        x_features = feature_matrix(out).to_numpy()
        x = np.hstack([x_species, x_features])

        log_density = model.predict_log_density(x)
        out["plausibility"] = normalize_log_density(log_density, model)
        frames.append(out[["h3", "date", "species", "plausibility"]])

    return pd.concat(frames, ignore_index=True)


def build_product_plausibility(
    batch: pd.DataFrame,
    product_name: str,
    payload: dict,
) -> pd.DataFrame:
    """Predict plausibility for one product payload."""
    if payload.get("model_type") == "joint_species":
        return build_joint_plausibility(batch, payload)

    species = payload.get("species", product_name.upper())
    out = batch[["h3", "date"]].copy()
    out["species"] = species
    out["plausibility"] = predict_plausibility(batch, payload)

    return out


def add_support_by_species(
    df: pd.DataFrame,
    support_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add historical support to rows grouped by species."""
    frames = []

    for species, group in df.groupby("species", sort=False):
        support = support_tables.get(species)
        frames.append(add_support_columns(group, support))

    return pd.concat(frames, ignore_index=True)


def build_hybrid_product_predictions(
    batch: pd.DataFrame,
    product_name: str,
    ml_payload: dict,
    bayesian_payload: dict,
    support_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Predict one hybrid product payload pair."""
    ml_out = build_product_predictions(batch, product_name, ml_payload)

    out = ml_out.rename(
        columns={"species_use_log_pred": "species_use_ml_log_pred"}
    )

    if HYBRID_MODE == "presence_gate":
        plausibility = build_product_plausibility(
            batch,
            product_name,
            bayesian_payload,
        )
        out = out.merge(
            plausibility,
            on=["h3", "date", "species"],
            how="inner",
        )

        plausibility_values = out["plausibility"].to_numpy(dtype="float32")
        out["hybrid_alpha"] = presence_gate(plausibility_values, out["species"])
        out["species_use_log_pred"] = apply_presence_gate(
            out["species_use_ml_log_pred"].to_numpy(dtype="float32"),
            plausibility_values,
            out["species"],
        )

        return out

    bayesian_out = build_product_predictions(
        batch,
        product_name,
        bayesian_payload,
    )
    bayesian_values = bayesian_out[
        ["h3", "date", "species", "species_use_log_pred"]
    ].rename(
        columns={
            "species_use_log_pred": "species_use_bayesian_log_pred",
        }
    )

    out = out.merge(
        bayesian_values,
        on=["h3", "date", "species"],
        how="inner",
    )
    out = add_support_by_species(out, support_tables)

    alpha, hybrid_pred = blend_hybrid_predictions(
        out["species_use_ml_log_pred"].to_numpy(dtype="float32"),
        out["species_use_bayesian_log_pred"].to_numpy(dtype="float32"),
        out["support_count"].to_numpy(dtype="float32"),
    )

    out["hybrid_alpha"] = alpha
    out["species_use_log_pred"] = hybrid_pred

    return out


def build_hybrid_species_predictions(
    batch: pd.DataFrame,
    ml_payloads: dict[str, dict],
    bayesian_payloads: dict[str, dict],
    support_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = []

    for species, ml_payload in ml_payloads.items():
        out = batch.copy()
        out["species"] = species

        support = support_tables.get(species)
        out = add_support_columns(out, support)

        bayesian_payload = bayesian_payloads[species]

        if HYBRID_MODE == "presence_gate":
            ml_pred = predict_species(out, ml_payload)
            plausibility = predict_plausibility(out, bayesian_payload)

            out["hybrid_alpha"] = plausibility
            out["species_use_ml_log_pred"] = ml_pred
            out["plausibility"] = plausibility
            out["species_use_log_pred"] = apply_presence_gate(
                ml_pred,
                plausibility,
            )
            frames.append(out)
            continue

        ml_pred, bayesian_pred = predict_hybrid_species(
            out,
            ml_payload,
            bayesian_payload,
        )

        alpha, hybrid_pred = blend_hybrid_predictions(
            ml_pred,
            bayesian_pred,
            out["support_count"].to_numpy(dtype="float32"),
        )

        out["hybrid_alpha"] = alpha
        out["species_use_ml_log_pred"] = ml_pred
        out["species_use_bayesian_log_pred"] = bayesian_pred
        out["species_use_log_pred"] = hybrid_pred
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def add_risk_columns(expanded: pd.DataFrame) -> pd.DataFrame:
    risk_log = (
        expanded["species_use_log_pred"]
        + expanded["fishing_activity_log"]
    )

    expanded["risk_log_pred"] = risk_log.astype("float32")

    return expanded


def output_columns(model_name: str) -> list[str]:
    cols = [
        "h3",
        "date",
        "species",
        "species_use_log_pred",
        "fishing_activity_log",
        "risk_log_pred",
    ]

    if model_name == "hybrid":
        cols.insert(4, "hybrid_alpha")
        cols.insert(5, "species_use_ml_log_pred")
        if HYBRID_MODE == "presence_gate":
            cols.insert(6, "plausibility")
        else:
            cols.insert(6, "species_use_bayesian_log_pred")

    return cols


def load_observed_fishing(year: int) -> pd.DataFrame:
    path = (
        paths["data"]
        / "modeling"
        / "fishing_training"
        / f"year={year}"
        / "part.parquet"
    )

    if not path.exists():
        return pd.DataFrame(columns=["h3", "date", FISHING_TARGET])

    df = pd.read_parquet(path)
    return df[["h3", "date", FISHING_TARGET]].copy()


# NEW FUNCTION: Load prediction grid partition for a given year
def load_prediction_grid(year: int) -> pd.DataFrame:
    """Load one modeling feature grid partition."""
    return load_feature_grid(year)


def merge_year_parts(model_name: str, year: int) -> Path:
    out_dir = output_dir(model_name, year)
    parts = sorted(out_dir.glob("part-*.parquet"))

    if not parts:
        raise FileNotFoundError(f"No chunks for year {year}")

    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)

    out_file = merged_output_path(model_name, year)
    df.to_parquet(out_file, index=False, compression="zstd")

    for p in parts:
        p.unlink()

    return out_file


def merge_product_year_parts(
    model_name: str,
    product_name: str,
    year: int,
) -> Path:
    out_dir = product_output_dir(model_name, product_name, year)
    parts = sorted(out_dir.glob("part-*.parquet"))

    if not parts:
        raise FileNotFoundError(
            f"No chunks for {prediction_model_name(model_name)}/"
            f"{product_name} year {year}"
        )

    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)

    out_file = product_merged_output_path(model_name, product_name, year)
    df.to_parquet(out_file, index=False, compression="zstd")

    for p in parts:
        p.unlink()

    return out_file


def prepare_prediction_base(year: int) -> pd.DataFrame:
    """Load feature grid once and add fishing exposure."""
    base = load_prediction_grid(year)

    if base.empty:
        print(f"Skipping {year}: feature_grid partition not found or empty")
        return base

    missing = [c for c in FEATURES if c not in base.columns]
    if missing:
        raise ValueError(f"Missing features in feature_grid: {missing}")

    base = drop_invalid_prediction_rows(base, year)

    if base.empty:
        print(f"Skipping {year}: no valid prediction rows after dropping NaNs")
        return base

    fishing = load_observed_fishing(year)

    base = base.merge(
        fishing,
        on=["h3", "date"],
        how="left",
    )

    base[FISHING_TARGET] = (
        base[FISHING_TARGET]
        .fillna(0.0)
        .astype("float32")
    )

    fishing_activity = base[FISHING_TARGET].to_numpy(dtype="float32")
    fishing_log = np.zeros_like(fishing_activity)
    positive = fishing_activity > 0
    fishing_log[positive] = np.log1p(fishing_activity[positive])

    base["fishing_activity"] = fishing_activity
    base["fishing_activity_log"] = fishing_log

    return base


def predict_year(
    model_name: str,
    year: int,
    species_payloads: dict[str, dict] | None = None,
    ml_payloads: dict[str, dict] | None = None,
    bayesian_payloads: dict[str, dict] | None = None,
    support_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    base = load_prediction_grid(year)

    if base.empty:
        print(f"Skipping {year}: feature_grid partition not found or empty")
        return

    missing = [c for c in FEATURES if c not in base.columns]
    if missing:
        raise ValueError(f"Missing features in feature_grid: {missing}")

    base = drop_invalid_prediction_rows(base, year)

    if base.empty:
        print(f"Skipping {year}: no valid prediction rows after dropping NaNs")
        return

    fishing = load_observed_fishing(year)

    base = base.merge(
        fishing,
        on=["h3", "date"],
        how="left",
    )

    base[FISHING_TARGET] = (
        base[FISHING_TARGET]
        .fillna(0.0)
        .astype("float32")
    )

    clean_output_dir(model_name, year)
    out_dir = output_dir(model_name, year)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(iter_batches(base, BATCH_ROWS)):
        fishing_activity = batch[FISHING_TARGET].to_numpy(dtype="float32")

        fishing_log = np.zeros_like(fishing_activity)
        positive = fishing_activity > 0
        fishing_log[positive] = np.log1p(fishing_activity[positive])

        batch = batch.copy()
        batch["fishing_activity"] = fishing_activity
        batch["fishing_activity_log"] = fishing_log

        if model_name == "hybrid":
            if support_tables is None:
                raise ValueError("Hybrid prediction requires support tables")

            expanded = build_hybrid_species_predictions(
                batch,
                ml_payloads,
                bayesian_payloads,
                support_tables,
            )
        else:
            expanded = build_species_predictions(batch, species_payloads)

        expanded = add_risk_columns(expanded)
        out = expanded[output_columns(model_name)]

        out_file = out_dir / f"part-{i:05d}.parquet"
        out.to_parquet(out_file, index=False, compression="zstd")

        print(f"Saved: {out_file} | Rows: {len(out)}")

    merged = merge_year_parts(model_name, year)
    print(f"Merged: {merged}")


def predict_product_year(
    base: pd.DataFrame,
    model_name: str,
    product_name: str,
    payload: dict,
    year: int,
) -> None:
    """Predict one product using a pre-loaded feature grid."""
    clean_product_output_dir(model_name, product_name, year)
    out_dir = product_output_dir(model_name, product_name, year)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(iter_batches(base, BATCH_ROWS)):
        expanded = build_product_predictions(batch, product_name, payload)
        expanded = add_risk_columns(expanded)
        out = expanded[output_columns(model_name)]

        out_file = out_dir / f"part-{i:05d}.parquet"
        out.to_parquet(out_file, index=False, compression="zstd")

        print(f"Saved: {out_file} | Rows: {len(out)}")

    merged = merge_product_year_parts(model_name, product_name, year)
    print(f"Merged: {merged}")


def predict_hybrid_product_year(
    base: pd.DataFrame,
    model_name: str,
    product_name: str,
    ml_payload: dict,
    bayesian_payload: dict,
    support_tables: dict[str, pd.DataFrame],
    year: int,
) -> None:
    """Predict one hybrid product using a pre-loaded feature grid."""
    clean_product_output_dir(model_name, product_name, year)
    out_dir = product_output_dir(model_name, product_name, year)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(iter_batches(base, BATCH_ROWS)):
        expanded = build_hybrid_product_predictions(
            batch,
            product_name,
            ml_payload,
            bayesian_payload,
            support_tables,
        )
        expanded = add_risk_columns(expanded)
        out = expanded[output_columns("hybrid")]

        out_file = out_dir / f"part-{i:05d}.parquet"
        out.to_parquet(out_file, index=False, compression="zstd")

        print(f"Saved: {out_file} | Rows: {len(out)}")

    merged = merge_product_year_parts(model_name, product_name, year)
    print(f"Merged: {merged}")


def selected_model_names() -> list[str]:
    """Return model names selected for prediction."""
    if PREDICTION_MODE == "single":
        return [ACTIVE_MODEL_NAME]

    if PREDICTION_MODE == "all":
        return list(MODEL_NAMES)

    if PREDICTION_MODE == "hybrid":
        return ["hybrid"]

    raise ValueError(f"Unknown PREDICTION_MODE: {PREDICTION_MODE}")


def predict_models() -> None:
    model_names = selected_model_names()

    if model_names == ["hybrid"]:
        model_name = "hybrid"
        print(f"\n=== Predicting with model: {prediction_model_name(model_name)} ===")
        ml_payloads = load_product_payloads(HYBRID_ML_MODEL_NAME)
        bayesian_payloads = load_product_payloads(HYBRID_BAYESIAN_MODEL_NAME)
        support_tables = load_hybrid_support()

        for product_name in PREDICTION_PRODUCTS:
            print(f"Prediction product: {product_name}")

            for year in available_years("environmental"):
                print(f"\n=== Loading feature_grid once for year {year} ===")
                base = prepare_prediction_base(year)

                if base.empty:
                    continue

                predict_hybrid_product_year(
                    base=base,
                    model_name=model_name,
                    product_name=product_name,
                    ml_payload=ml_payloads[product_name],
                    bayesian_payload=bayesian_payloads[product_name],
                    support_tables=support_tables,
                    year=year,
                )

        return

    for model_name in model_names:
        print(f"\n=== Predicting with model: {prediction_model_name(model_name)} ===")
        payloads = load_product_payloads(model_name)

        for product_name in PREDICTION_PRODUCTS:
            print(f"Prediction product: {product_name}")

            for year in available_years("environmental"):
                print(f"\n=== Loading feature_grid once for year {year} ===")
                base = prepare_prediction_base(year)

                if base.empty:
                    continue

                predict_product_year(
                    base=base,
                    model_name=model_name,
                    product_name=product_name,
                    payload=payloads[product_name],
                    year=year,
                )
