"""Train species-use models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, SPECIES_TARGET


RANDOM_STATE = 42

MODEL_NAMES = [
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
]

PRIMARY_MODEL_NAME = "extra_trees"

MAX_ZERO_ROWS = 250_000
MAX_POSITIVE_ROWS = None

MODEL_DIR = paths["data"] / "modeling" / "models"
METRICS_DIR = paths["data"] / "modeling" / "metrics"


def load_table(name: str) -> pd.DataFrame:
    """Load partitioned modeling table."""
    root = paths["data"] / "modeling" / name
    frames = []

    for path in sorted(root.glob("year=*/part.parquet")):
        frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No data found for {name}")

    return pd.concat(frames, ignore_index=True)


def split_random(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows randomly into train and test sets."""
    test = df.sample(
        frac=test_fraction,
        random_state=RANDOM_STATE,
    )

    train = df.drop(test.index).copy()
    test = test.copy()

    return train, test


def metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def prepare_species_table() -> pd.DataFrame:
    """Load and prepare species training table."""
    df = load_table("species_training")

    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols).copy()

    df["_y"] = np.log1p(df[SPECIES_TARGET])
    # df["_y"] = df[SPECIES_TARGET]

    return df


# def sample_training_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """Sample zeros while preserving positive rows."""
#     positive = df[df[SPECIES_TARGET] > 0].copy()
#     zero = df[df[SPECIES_TARGET] == 0].copy()

#     if MAX_POSITIVE_ROWS is not None and len(positive) > MAX_POSITIVE_ROWS:
#         positive = positive.sample(
#             n=MAX_POSITIVE_ROWS,
#             random_state=RANDOM_STATE,
#         )

#     if MAX_ZERO_ROWS is not None and len(zero) > MAX_ZERO_ROWS:
#         zero = zero.sample(
#             n=MAX_ZERO_ROWS,
#             random_state=RANDOM_STATE,
#         )

#     out = pd.concat([positive, zero], ignore_index=True)

#     return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

def sample_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Balance zeros vs positives."""
    positive = df[df[SPECIES_TARGET] > 0].copy()
    zero = df[df[SPECIES_TARGET] == 0].copy()

    n_pos = len(positive)

    zero = zero.sample(
        n=min(n_pos, len(zero)),
        random_state=RANDOM_STATE,
    )

    out = pd.concat([positive, zero], ignore_index=True)

    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

def build_model(model_name: str):
    """Build model by name."""
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {model_name}")


def sample_weights(y_train: pd.Series) -> np.ndarray:
    """Return sample weights from raw target scale."""
    raw_target = np.expm1(y_train)
    # weights = 1.0 + raw_target
    # weights = 1.0 + 2.0 * raw_target
    weights = 1 + raw_target ** 0.75

    return weights.to_numpy(dtype="float32")


def save_payload(payload: dict, path: Path) -> None:
    """Save model payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def evaluate_predictions(y_test: pd.Series, pred_log: np.ndarray) -> dict:
    """Evaluate predictions on raw target scale."""
    pred = np.expm1(pred_log)
    pred = np.maximum(pred, 0.0)

    y_test_raw = np.expm1(y_test)

    return metrics(y_test_raw, pred)


def train_joint_species_model(
    df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Train joint species model."""
    train, test = split_random(df)

    x_train_num = train[FEATURES].to_numpy()
    x_test_num = test[FEATURES].to_numpy()

    y_train = train["_y"]
    y_test = test["_y"]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    x_train_species = enc.fit_transform(train[["species"]])
    x_test_species = enc.transform(test[["species"]])

    x_train = np.hstack([x_train_species, x_train_num])
    x_test = np.hstack([x_test_species, x_test_num])

    model = build_model(model_name)

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights(y_train),
    )

    pred_log = model.predict(x_test)

    m = evaluate_predictions(y_test, pred_log)
    m["model"] = model_name
    m["model_type"] = "joint_species"
    m["species"] = "all"
    m["train_rows"] = int(len(train))
    m["test_rows"] = int(len(test))

    payload = {
        "model": model,
        "encoder": enc,
        "features": FEATURES,
        "target": SPECIES_TARGET,
        "log_target": True,
        "model_name": model_name,
        "model_type": "joint_species",
    }

    save_payload(
        payload,
        MODEL_DIR / f"species_model_joint_{model_name}.joblib",
    )

    print("Joint species model:", model_name)
    print(m)

    return m


def train_single_species_model(
    species_name: str,
    df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Train one species-use model."""
    species_df = df[df["species"] == species_name].copy()

    if species_df.empty:
        raise ValueError(f"No rows found for species: {species_name}")

    train, test = split_random(species_df)

    x_train = train[FEATURES]
    x_test = test[FEATURES]

    y_train = train["_y"]
    y_test = test["_y"]

    model = build_model(model_name)

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights(y_train),
    )

    pred_log = model.predict(x_test)

    m = evaluate_predictions(y_test, pred_log)
    m["model"] = model_name
    m["model_type"] = "single_species"
    m["species"] = species_name
    m["train_rows"] = int(len(train))
    m["test_rows"] = int(len(test))

    safe_name = species_name.lower()

    payload = {
        "model": model,
        "species": species_name,
        "features": FEATURES,
        "target": SPECIES_TARGET,
        "log_target": True,
        "model_name": model_name,
        "model_type": "single_species",
    }

    save_payload(
        payload,
        MODEL_DIR / f"species_model_{safe_name}_{model_name}.joblib",
    )

    print(f"{species_name} species model:", model_name)
    print(m)

    return m


def save_metrics(rows: list[dict]) -> None:
    """Save model comparison metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    path = METRICS_DIR / "species_model_comparison.csv"

    df.to_csv(path, index=False)

    print("Metrics saved:", path)


def train_models() -> None:
    """Train species models."""
    df = prepare_species_table()
    df = sample_training_rows(df)

    species_values = sorted(df["species"].dropna().unique().tolist())

    rows = []

    for model_name in MODEL_NAMES:
        rows.append(train_joint_species_model(df, model_name))

        for species_name in species_values:
            rows.append(
                train_single_species_model(
                    species_name=species_name,
                    df=df,
                    model_name=model_name,
                )
            )

    save_metrics(rows)