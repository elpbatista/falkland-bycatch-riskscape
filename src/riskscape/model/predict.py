"""Generate model predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import (
    FEATURES,
    available_years,
    join_features,
    load_partition,
    load_static,
    species_list,
)


BATCH_ROWS = 250_000

MODEL_DIR = paths["data"] / "modeling" / "models"
PREDICTION_ROOT = paths["data"] / "modeling" / "predictions"


def load_model_payload(path: Path):
    """Load model payload."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    return joblib.load(path)


def output_dir(year: int) -> Path:
    """Return yearly prediction output directory."""
    return PREDICTION_ROOT / f"year={year}"


def clean_output_dir(year: int) -> None:
    """Remove existing prediction chunks for one year."""
    out_dir = output_dir(year)

    if not out_dir.exists():
        return

    for path in out_dir.glob("part-*.parquet"):
        path.unlink()


def iter_batches(df: pd.DataFrame, batch_rows: int):
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def get_payload_model(payload):
    """Return model object from payload."""
    if isinstance(payload, dict):
        return payload["model"]

    return payload


def predict_fishing(
    batch: pd.DataFrame,
    fishing_payload,
) -> np.ndarray:
    """Predict fishing activity for base h3/date rows."""
    model = get_payload_model(fishing_payload)

    pred = model.predict(batch[FEATURES])
    pred = np.maximum(pred, 0.0)

    return pred.astype("float32")


def predict_species(
    expanded: pd.DataFrame,
    species_payload,
) -> np.ndarray:
    """Predict species use for h3/date/species rows."""
    model = species_payload["model"]
    encoder = species_payload["encoder"]

    species_encoded = encoder.transform(expanded[["species"]])
    feature_values = expanded[FEATURES].to_numpy()

    x = np.hstack([species_encoded, feature_values])

    pred = model.predict(x)
    pred = np.maximum(pred, 0.0)

    return pred.astype("float32")


def predict_year(
    year: int,
    static: pd.DataFrame,
    species_df: pd.DataFrame,
    species_payload,
    fishing_payload,
) -> None:
    """Generate predictions for one year."""
    env = load_partition("environmental", year)

    if env.empty:
        return

    base = env[["h3", "date"]].copy()
    base = join_features(base, env, static)

    clean_output_dir(year)
    out_dir = output_dir(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_species = len(species_df)

    for i, batch in enumerate(iter_batches(base, BATCH_ROWS)):
        fishing_pred = predict_fishing(batch, fishing_payload)

        # expanded = batch[["h3", "date"]].merge(species_df, how="cross")
        expanded = batch.merge(species_df, how="cross")
        species_pred = predict_species(expanded, species_payload)

        expanded["species_use_pred"] = species_pred
        expanded["fishing_activity_pred"] = np.repeat(fishing_pred, n_species)
        expanded["risk_pred"] = (
            expanded["species_use_pred"]
            * expanded["fishing_activity_pred"]
        ).astype("float32")

        out = expanded[
            [
                "h3",
                "date",
                "species",
                "species_use_pred",
                "fishing_activity_pred",
                "risk_pred",
            ]
        ]

        out_file = out_dir / f"part-{i:05d}.parquet"
        out.to_parquet(out_file, index=False, compression="zstd")

        print(f"Saved: {out_file}")
        print(f"Rows: {len(out)}")


def predict_models() -> None:
    """Generate full-grid model predictions."""
    static = load_static()
    species_df = pd.DataFrame({"species": species_list()})

    species_payload = load_model_payload(MODEL_DIR / "species_model.joblib")
    fishing_payload = load_model_payload(MODEL_DIR / "fishing_model.joblib")

    for year in available_years("environmental"):
        predict_year(
            year=year,
            static=static,
            species_df=species_df,
            species_payload=species_payload,
            fishing_payload=fishing_payload,
        )