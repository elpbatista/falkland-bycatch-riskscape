"""Assign dominant Bayesian/GMM ecological components."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.block_cv_train import SplitSummary
from riskscape.model.dataset import load_feature_grid, modeling_root
from riskscape.utils.dates import normalize_date_column


DEFAULT_MODEL_PATH = (
    paths["data"]
    / "modeling"
    / "models"
    / "bayesian_gmm_k30"
    / "species_model_joint.joblib"
)

DEFAULT_OUTPUT_TABLE = "cube_components_bayesian_gmm_k30"
DEFAULT_OUT_ROOT = modeling_root(DEFAULT_OUTPUT_TABLE)

BATCH_ROWS = 250_000
LOG_2PI = np.log(2.0 * np.pi)


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


def component_path(out_root: Path, year: int) -> Path:
    """Return component assignment output partition path."""
    return out_root / f"year={year}" / "part.parquet"


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


def component_entropy(proba: np.ndarray) -> np.ndarray:
    """Compute normalized component entropy."""
    eps = 1e-12

    entropy = -np.sum(
        proba * np.log(proba + eps),
        axis=1,
    )

    max_entropy = np.log(proba.shape[1])

    return (entropy / max_entropy).astype("float32")


def feature_offset(payload: dict) -> int:
    """Return the number of species-encoder columns in the fitted joint model."""
    encoder = payload["encoder"]

    return int(sum(len(categories) for categories in encoder.categories_))


def scaled_feature_matrix(
    batch: pd.DataFrame,
    payload: dict,
) -> np.ndarray:
    """Return environmental features scaled with the fitted joint-model scaler."""
    model = payload["model"]
    offset = feature_offset(payload)
    x = feature_matrix(batch, payload).to_numpy(dtype="float64")
    mean = model.scaler.mean_[offset:]
    scale = model.scaler.scale_[offset:]

    return (x - mean) / scale


def component_log_probability(
    x_scaled: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Return multivariate Gaussian log probability for one component."""
    jitter = 1e-6
    covariance = covariance.copy()

    for _ in range(5):
        try:
            chol = np.linalg.cholesky(covariance)
            break
        except np.linalg.LinAlgError:
            covariance += np.eye(covariance.shape[0]) * jitter
            jitter *= 10.0
    else:
        chol = np.linalg.cholesky(
            covariance + np.eye(covariance.shape[0]) * jitter
        )

    centered = (x_scaled - mean).T
    solved = np.linalg.solve(chol, centered)
    quadratic = np.sum(solved * solved, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    n_features = x_scaled.shape[1]

    return -0.5 * (n_features * LOG_2PI + log_det + quadratic)


def environmental_component_probabilities(
    batch: pd.DataFrame,
    payload: dict,
) -> np.ndarray:
    """Return component probabilities from environmental feature values only."""
    model = payload["model"]
    offset = feature_offset(payload)
    x_scaled = scaled_feature_matrix(batch, payload)
    means = model.gmm.means_[:, offset:]
    covariances = model.gmm.covariances_[:, offset:, offset:]
    log_weights = np.log(model.gmm.weights_)
    log_probs = np.empty((len(batch), model.gmm.n_components), dtype="float64")

    for component in range(model.gmm.n_components):
        log_probs[:, component] = (
            log_weights[component]
            + component_log_probability(
                x_scaled,
                means[component],
                covariances[component],
            )
        )

    max_log = log_probs.max(axis=1, keepdims=True)
    shifted = np.exp(log_probs - max_log)

    return shifted / shifted.sum(axis=1, keepdims=True)


def process_batch(
    batch: pd.DataFrame,
    payload: dict,
) -> pd.DataFrame:
    """Assign dominant ecological component from environmental features."""
    proba = environmental_component_probabilities(batch, payload)

    component = proba.argmax(axis=1)

    dominant_probability = proba.max(axis=1)

    entropy = component_entropy(proba)

    out = batch[
        [
            "h3",
            "date",
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


def assign_year(payload: dict, year: int, out_root: Path) -> Path:
    """Assign components for one feature-grid year."""
    feature_grid = drop_invalid_feature_rows(
        load_model_feature_grid(payload, year),
        payload,
    )

    frames = []

    for batch in iter_batches(feature_grid, BATCH_ROWS):
        frames.append(
            process_batch(batch, payload)
        )

    out = normalize_date_column(pd.concat(frames, ignore_index=True))
    out_path = component_path(out_root, year)

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assign Bayesian/GMM components from environmental features.",
    )
    parser.add_argument(
        "--year",
        type=int,
        action="append",
        help="Year to process. Can be provided more than once. Defaults to all years.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the Bayesian/GMM joint species model payload.",
    )
    parser.add_argument(
        "--output-table",
        default=DEFAULT_OUTPUT_TABLE,
        help=(
            "Modeling table name for outputs. Defaults to the selected "
            "Bayesian GMM k30 component table."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run component assignment."""
    args = parse_args()
    setattr(sys.modules["__main__"], "SplitSummary", SplitSummary)
    payload = joblib.load(args.model_path)
    out_root = modeling_root(args.output_table)
    years = args.year if args.year else component_years()

    print(f"Model: {args.model_path}")
    print(f"Output table: {out_root}")

    for year in years:
        print()
        print(f"=== Assigning components for {year} ===")
        assign_year(payload, year, out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
