"""Train baseline models."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, FISHING_TARGET, SPECIES_TARGET


RANDOM_STATE = 42
N_ESTIMATORS = 100

MAX_FISHING_TRAIN_ROWS = 300_000
MAX_FISHING_TEST_ROWS = 300_000

MODEL_DIR = paths["data"] / "modeling" / "models"


def load_table(name: str) -> pd.DataFrame:
    """Load partitioned modeling table."""
    root = paths["data"] / "modeling" / name
    frames = []

    for path in sorted(root.glob("year=*/part.parquet")):
        frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No data found for {name}")

    return pd.concat(frames, ignore_index=True)


def add_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add year column from date."""
    out = df.copy()
    out["year"] = out["date"].dt.year
    return out


def split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use last available year as test set."""
    df = add_year(df)
    years = sorted(df["year"].unique())

    if len(years) < 2:
        raise ValueError("At least two years are required for time split")

    test_year = years[-1]

    train = df[df["year"] < test_year].copy()
    test = df[df["year"] == test_year].copy()

    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test set")

    return train, test


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sample rows if dataframe exceeds max_rows."""
    if len(df) <= max_rows:
        return df

    return df.sample(
        n=max_rows,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_species_model() -> None:
    """Train species-use model."""
    df = load_table("species_training")

    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols)

    train, test = split_time(df)

    x_train = train[FEATURES]
    x_test = test[FEATURES]

    y_train = train[SPECIES_TARGET]
    y_test = test[SPECIES_TARGET]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    species_train = enc.fit_transform(train[["species"]])
    species_test = enc.transform(test[["species"]])

    x_train = np.hstack([species_train, x_train.values])
    x_test = np.hstack([species_test, x_test.values])

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    m = metrics(y_test, pred)
    m["train_rows"] = int(len(train))
    m["test_rows"] = int(len(test))

    path = MODEL_DIR / "species_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "encoder": enc,
            "features": FEATURES,
            "target": SPECIES_TARGET,
        },
        path,
    )

    print("Species model saved:", path)
    print(m)


def train_fishing_model() -> None:
    """Train fishing-activity model."""
    df = load_table("fishing_training")

    cols = [FISHING_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols)

    train, test = split_time(df)

    train = sample_rows(train, MAX_FISHING_TRAIN_ROWS)
    test = sample_rows(test, MAX_FISHING_TEST_ROWS)

    x_train = train[FEATURES]
    x_test = test[FEATURES]

    y_train = train[FISHING_TARGET]
    y_test = test[FISHING_TARGET]

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    m = metrics(y_test, pred)
    m["train_rows"] = int(len(train))
    m["test_rows"] = int(len(test))

    path = MODEL_DIR / "fishing_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": FISHING_TARGET,
        },
        path,
    )

    print("Fishing model saved:", path)
    print(m)


def train_models() -> None:
    """Train all baseline models."""
    train_species_model()
    train_fishing_model()