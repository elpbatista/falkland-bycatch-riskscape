"""Calculate Bayesian GMM environmental plausibility surfaces."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, available_years, species_list


MODEL_NAME = "bayesian_gmm"
BATCH_ROWS = 250_000

MODEL_DIR = paths["data"] / "modeling" / "models" / MODEL_NAME
FEATURE_GRID_ROOT = paths["data"] / "modeling" / "feature_grid"
OUT_ROOT = paths["data"] / "modeling" / "plausibility" / MODEL_NAME


def load_payload(path: Path) -> dict:
    """Load model payload."""
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    return joblib.load(path)


def load_feature_grid(year: int) -> pd.DataFrame:
    """Load feature grid for one year."""
    path = FEATURE_GRID_ROOT / f"year={year}" / "part.parquet"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def iter_batches(df: pd.DataFrame, batch_rows: int):
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid feature matrix."""
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if x.isna().any().any():
        missing = x.columns[x.isna().any()].tolist()
        raise ValueError(f"NaN values in feature grid: {missing}")

    return x


def drop_invalid_feature_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Drop rows with invalid feature values."""
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    mask = out[FEATURES].isna().any(axis=1)

    if not mask.any():
        return out

    print(f"Dropping {int(mask.sum())} invalid feature rows for {year}")

    return out.loc[~mask].reset_index(drop=True)


def normalize_log_density(log_density: np.ndarray, model) -> np.ndarray:
    """Normalize log-density to plausibility range 0-1."""
    denom = model.max_log_density - model.min_log_density

    if denom <= 0:
        return np.zeros_like(log_density, dtype="float32")

    plausibility = (log_density - model.min_log_density) / denom
    plausibility = np.clip(plausibility, 0.0, 1.0)

    return plausibility.astype("float32")


def save_year(df: pd.DataFrame, product: str, year: int) -> Path:
    """Save plausibility partition."""
    out_dir = OUT_ROOT / product / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "part.parquet"
    df.to_parquet(path, index=False, compression="zstd")

    return path


def calculate_single_species(
    base: pd.DataFrame,
    species: str,
    payload: dict,
) -> pd.DataFrame:
    """Calculate plausibility for one single-species model."""
    model = payload["model"]

    log_density = model.predict_log_density(feature_matrix(base))
    plausibility = normalize_log_density(log_density, model)

    out = base[["h3", "date"]].copy()
    out["species"] = species
    out["log_density"] = log_density.astype("float32")
    out["plausibility"] = plausibility

    return out


def calculate_joint_species(
    base: pd.DataFrame,
    species: str,
    payload: dict,
) -> pd.DataFrame:
    """Calculate plausibility for one species using the joint model."""
    model = payload["model"]
    encoder = payload.get("encoder")

    if encoder is None:
        raise KeyError("Joint payload is missing encoder")

    x_species = encoder.transform(
        pd.DataFrame({"species": [species] * len(base)})
    )
    x_features = feature_matrix(base).to_numpy()
    x = np.hstack([x_species, x_features])

    log_density = model.predict_log_density(x)
    plausibility = normalize_log_density(log_density, model)

    out = base[["h3", "date"]].copy()
    out["species"] = species
    out["log_density"] = log_density.astype("float32")
    out["plausibility"] = plausibility

    return out


def calculate_product_year(
    base: pd.DataFrame,
    year: int,
    product: str,
    payload: dict,
    species_values: list[str],
) -> pd.DataFrame:
    """Calculate one plausibility product for one year."""
    frames = []

    if product == "joint":
        for batch in iter_batches(base, BATCH_ROWS):
            batch_frames = [
                calculate_joint_species(batch, species, payload)
                for species in species_values
            ]
            frames.append(pd.concat(batch_frames, ignore_index=True))
    else:
        for batch in iter_batches(base, BATCH_ROWS):
            frames.append(calculate_single_species(batch, product.upper(), payload))

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    """Run plausibility calculation."""
    species_values = species_list()

    joint_payload = load_payload(MODEL_DIR / "species_model_joint.joblib")

    for year in available_years("environmental"):
        base = load_feature_grid(year)

        if base.empty:
            print(f"Skipping {year}: missing feature_grid")
            continue

        base = drop_invalid_feature_rows(base, year)

        products = {"joint": joint_payload}

        for product, payload in products.items():
            out = calculate_product_year(
                base=base,
                year=year,
                product=product,
                payload=payload,
                species_values=species_values,
            )

            path = save_year(out, product, year)
            print(f"Saved: {path}")
            print(f"Rows: {len(out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
