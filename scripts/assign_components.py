"""Assign dominant Bayesian/GMM ecological components."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import load_feature_grid, modeling_root


MODEL_PATH = (
    paths["data"]
    / "modeling"
    / "models"
    / "bayesian_gmm"
    / "species_model_joint.joblib"
)

OUT_ROOT = modeling_root("cube_components")

BATCH_ROWS = 250_000


def model_features(payload: dict) -> list[str]:
    """Return the feature columns saved with the trained model."""
    features = payload.get("features")

    if not features:
        raise KeyError("Model payload is missing training feature list")

    return list(features)


def model_species(payload: dict) -> list[str]:
    """Return species known by the joint model encoder."""
    encoder = payload.get("encoder")

    if encoder is None:
        raise KeyError("Joint model payload is missing encoder")

    return encoder.categories_[0].tolist()


def component_path(year: int) -> Path:
    """Return component assignment output partition path."""
    return OUT_ROOT / f"year={year}" / "part.parquet"


def component_years() -> list[int]:
    """Return years with feature-grid partitions."""
    years = []

    for path in sorted(modeling_root("feature_grid").glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=")[1]))

    if not years:
        raise FileNotFoundError("No feature_grid partitions found")

    return years


def load_model_feature_grid(payload: dict, year: int) -> pd.DataFrame:
    """Load h3/date feature grid columns required by the trained model."""
    feature_grid = load_feature_grid(year)

    if feature_grid.empty:
        raise FileNotFoundError(f"Feature grid is empty for year {year}")

    keep = ["h3", "date"] + model_features(payload)

    return feature_grid[keep].copy()


def iter_batches(df: pd.DataFrame, batch_rows: int):
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def feature_matrix(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Return valid feature matrix using the model's training features."""
    features = model_features(payload)
    missing = [col for col in features if col not in batch.columns]

    if missing:
        raise KeyError(f"Input is missing model feature columns: {missing}")

    x = batch[features].replace([np.inf, -np.inf], np.nan)

    if x.isna().any().any():
        invalid = x.columns[x.isna().any()].tolist()
        raise ValueError(f"NaN values in model feature columns: {invalid}")

    return x


def drop_invalid_feature_rows(
    df: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Drop rows with missing or infinite model feature values."""
    features = model_features(payload)
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    mask = out[features].isna().any(axis=1)

    if not mask.any():
        return out

    missing_counts = out.loc[mask, features].isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )

    print(f"Dropping {int(mask.sum())} rows with invalid model features")
    print(missing_counts)

    return out.loc[~mask].reset_index(drop=True)


def build_joint_matrix(
    batch: pd.DataFrame,
    payload: dict,
) -> np.ndarray:
    """Build joint-species feature matrix."""
    encoder = payload["encoder"]

    x_species = encoder.transform(
        batch[["species"]]
    )

    x_features = feature_matrix(batch, payload).to_numpy()

    return np.hstack([x_species, x_features])


def add_species_rows(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Expand one feature-grid batch to all species known by the model."""
    frames = []

    for species in model_species(payload):
        out = batch.copy()
        out["species"] = species
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def component_entropy(proba: np.ndarray) -> np.ndarray:
    """Compute normalized component entropy."""
    eps = 1e-12

    entropy = -np.sum(
        proba * np.log(proba + eps),
        axis=1,
    )

    max_entropy = np.log(proba.shape[1])

    return (entropy / max_entropy).astype("float32")


def process_batch(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Assign dominant ecological component."""
    model = payload["model"]

    x = build_joint_matrix(batch, payload)

    x_scaled = model.scaler.transform(x)

    component = model.gmm.predict(x_scaled)

    proba = model.gmm.predict_proba(x_scaled)

    dominant_probability = proba.max(axis=1)

    entropy = component_entropy(proba)

    out = batch[
        [
            "h3",
            "date",
            "species",
        ]
    ].copy()

    out["component"] = component.astype("int16")

    out["component_probability"] = (
        dominant_probability.astype("float32")
    )

    out["component_entropy"] = entropy

    for i in range(proba.shape[1]):
        out[f"component_{i}_probability"] = (
            proba[:, i].astype("float32")
        )

    return out


def assign_year(payload: dict, year: int) -> Path:
    """Assign components for one feature-grid year."""
    feature_grid = drop_invalid_feature_rows(
        load_model_feature_grid(payload, year),
        payload,
    )

    frames = []

    for batch in iter_batches(feature_grid, BATCH_ROWS):
        batch = add_species_rows(batch, payload)
        frames.append(
            process_batch(batch, payload)
        )

    out = pd.concat(frames, ignore_index=True)
    out_path = component_path(year)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_parquet(
        out_path,
        index=False,
        compression="zstd",
    )

    print(out.head())

    print()
    print("Component counts:")
    print(
        out["component"]
        .value_counts()
        .sort_index()
    )

    print()
    print(f"Saved: {out_path}")

    return out_path


def main() -> int:
    """Run component assignment."""
    payload = joblib.load(MODEL_PATH)

    for year in component_years():
        print()
        print(f"=== Assigning components for {year} ===")
        assign_year(payload, year)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
