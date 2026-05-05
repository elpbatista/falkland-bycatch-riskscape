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
    available_years,
    join_features,
    load_partition,
    load_static,
    species_list,
)

from riskscape.model.train import PRIMARY_MODEL_NAME

BATCH_ROWS = 250_000

MODEL_DIR = paths["data"] / "modeling" / "models"
PREDICTION_ROOT = paths["data"] / "modeling" / "predictions"


def load_model_payload(path: Path):
    """Load model payload."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    return joblib.load(path)


def species_model_path(species: str) -> Path:
    """Return selected single-species model path."""
    return (
        MODEL_DIR
        / f"species_model_{species.lower()}_{PRIMARY_MODEL_NAME}.joblib"
    )


def load_species_payloads(species_values: list[str]) -> dict[str, dict]:
    """Load one species model per species."""
    payloads = {}

    for species in species_values:
        path = species_model_path(species)
        payload = load_model_payload(path)
        payloads[species] = payload

        model_name = payload.get("model_name", "unknown")
        print(f"Loaded {species} model: {model_name}")

    return payloads


def observed_fishing_path(year: int) -> Path:
    """Return observed fishing modeling partition path."""
    return (
        paths["data"]
        / "modeling"
        / "fishing_training"
        / f"year={year}"
        / "part.parquet"
    )


def load_observed_fishing(year: int) -> pd.DataFrame:
    """Load observed fishing activity for one year."""
    path = observed_fishing_path(year)

    if not path.exists():
        return pd.DataFrame(columns=["h3", "date", FISHING_TARGET])

    df = pd.read_parquet(path)

    return df[["h3", "date", FISHING_TARGET]].copy()


def output_dir(year: int) -> Path:
    """Return yearly prediction output directory."""
    return PREDICTION_ROOT / f"year={year}"


def merged_output_path(year: int) -> Path:
    """Return merged yearly prediction output path."""
    return output_dir(year) / "part.parquet"


def clean_output_dir(year: int) -> None:
    """Remove existing prediction files for one year."""
    out_dir = output_dir(year)

    if not out_dir.exists():
        return

    for path in out_dir.glob("part-*.parquet"):
        path.unlink()

    merged = merged_output_path(year)
    if merged.exists():
        merged.unlink()


def iter_batches(df: pd.DataFrame, batch_rows: int):
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def validate_model_features(species_payload) -> None:
    """Validate model feature compatibility."""
    payload_features = species_payload.get("features", FEATURES)

    if list(payload_features) != list(FEATURES):
        raise ValueError(
            "Model features do not match current FEATURES. "
            f"Model: {list(payload_features)}. "
            f"Current: {list(FEATURES)}."
        )


def predict_species(
    batch: pd.DataFrame,
    species_payload,
) -> np.ndarray:
    """Predict species use in model target space."""
    validate_model_features(species_payload)

    model = species_payload["model"]

    pred = model.predict(batch[FEATURES])
    pred = np.maximum(pred, 0.0)

    return pred.astype("float32")


def merge_year_parts(year: int) -> Path:
    """Merge yearly prediction chunks into one partition."""
    out_dir = output_dir(year)
    parts = sorted(out_dir.glob("part-*.parquet"))

    if not parts:
        raise FileNotFoundError(f"No prediction chunks found for year {year}")

    frames = [pd.read_parquet(path) for path in parts]
    df = pd.concat(frames, ignore_index=True)

    out_file = merged_output_path(year)
    df.to_parquet(out_file, index=False, compression="zstd")

    for path in parts:
        path.unlink()

    return out_file


def build_species_predictions(
    batch: pd.DataFrame,
    species_payloads: dict[str, dict],
) -> pd.DataFrame:
    """Predict all species for one batch."""
    frames = []

    for species, payload in species_payloads.items():
        out = batch.copy()
        out["species"] = species
        out["species_use_log_pred"] = predict_species(out, payload)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def predict_year(
    year: int,
    static: pd.DataFrame,
    species_payloads: dict[str, dict],
) -> None:
    """Generate observed-fishing risk predictions for one year."""
    env = load_partition("environmental", year)

    if env.empty:
        return

    base = env[["h3", "date"]].copy()
    base = join_features(base, env, static)

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

    clean_output_dir(year)
    out_dir = output_dir(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(iter_batches(base, BATCH_ROWS)):
        fishing_activity = batch[FISHING_TARGET].to_numpy(dtype="float32")

        fishing_activity_log = np.zeros_like(fishing_activity, dtype="float32")
        positive = fishing_activity > 0

        fishing_activity_log[positive] = np.log1p(
            fishing_activity[positive]
        ).astype("float32")

        batch = batch.copy()
        batch["fishing_activity"] = fishing_activity
        batch["fishing_activity_log"] = fishing_activity_log

        expanded = build_species_predictions(batch, species_payloads)

        has_fishing = expanded["fishing_activity"] > 0
        risk_log = (
            expanded["species_use_log_pred"]
            + expanded["fishing_activity_log"]
        )

        expanded["risk_log_pred"] = risk_log.where(
            has_fishing,
            0.0,
        ).astype("float32")

        out = expanded[
            [
                "h3",
                "date",
                "species",
                "species_use_log_pred",   # hazard
                "fishing_activity_log",   # exposure
                "risk_log_pred",          # combined
                "fishing_activity",
            ]
        ]

        out_file = out_dir / f"part-{i:05d}.parquet"
        out.to_parquet(out_file, index=False, compression="zstd")

        print(f"Saved: {out_file}")
        print(f"Rows: {len(out)}")

    merged = merge_year_parts(year)
    print(f"Merged: {merged}")


def predict_models() -> None:
    """Generate full-grid historical risk predictions."""
    static = load_static()
    species_values = species_list()
    species_payloads = load_species_payloads(species_values)

    for year in available_years("environmental"):
        predict_year(
            year=year,
            static=static,
            species_payloads=species_payloads,
        )