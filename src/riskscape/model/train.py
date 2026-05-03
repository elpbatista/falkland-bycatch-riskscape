"""Train baseline models."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, FISHING_TARGET, SPECIES_TARGET


RANDOM_STATE = 42

SPECIES_N_ESTIMATORS = 300

FISHING_MAX_ITER = 300
FISHING_LEARNING_RATE = 0.05
FISHING_MAX_LEAF_NODES = 31

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

def split_random(df: pd.DataFrame, test_fraction: float = 0.25):
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

def sample_by_year(df: pd.DataFrame, max_rows_per_year: int) -> pd.DataFrame:
    """Sample up to max_rows_per_year from each year."""
    frames = []

    for _, group in df.groupby("year", sort=True):
        if len(group) > max_rows_per_year:
            group = group.sample(
                n=max_rows_per_year,
                random_state=RANDOM_STATE,
            )

        frames.append(group)

    return pd.concat(frames, ignore_index=True)

def train_species_model() -> None:
    """Train species-use model with Random Forest."""
    df = load_table("species_training")

    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols)

    # --- stabilize target ---
    df[SPECIES_TARGET] = df[SPECIES_TARGET].clip(upper=20)
    y = np.log1p(df[SPECIES_TARGET])

    df = df.copy()
    df["_y"] = y

    # It is not perfect, because random split can leak temporal similarity, 
    # but for a quick baseline improvement it is much better than testing only on one species

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

    model = RandomForestRegressor(
        n_estimators=SPECIES_N_ESTIMATORS,
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    pred = np.expm1(pred)
    pred = np.maximum(pred, 0.0)

    y_test_inv = np.expm1(y_test)

    m = metrics(y_test_inv, pred)
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
            "log_target": True,
        },
        path,
    )

    print("Species model saved:", path)
    print(m)


def train_fishing_model() -> None:
    """Train fishing-activity model with HistGradientBoosting."""
    df = load_table("fishing_training")

    cols = [FISHING_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols)

    # --- stabilize target ---
    y = np.log1p(df[FISHING_TARGET])

    df = df.copy()
    df["_y"] = y

    train, test = split_time(df)

    train = sample_by_year(train, 250_000)

    if len(test) > 1_000_000:
        test = test.sample(
            n=1_000_000,
            random_state=RANDOM_STATE,
        )    

    x_train = train[FEATURES]
    x_test = test[FEATURES]

    y_train = train["_y"]
    y_test = test["_y"]

    model = HistGradientBoostingRegressor(
        max_iter=FISHING_MAX_ITER,
        learning_rate=FISHING_LEARNING_RATE,
        max_leaf_nodes=FISHING_MAX_LEAF_NODES,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    pred = np.expm1(pred)
    pred = np.maximum(pred, 0.0)

    y_test_inv = np.expm1(y_test)

    m = metrics(y_test_inv, pred)
    m["train_rows"] = int(len(train))
    m["test_rows"] = int(len(test))

    path = MODEL_DIR / "fishing_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": FISHING_TARGET,
            "log_target": True,
        },
        path,
    )

    print("Fishing model saved:", path)
    print(m)


def train_models() -> None:
    """Train all baseline models."""
    train_species_model()
    # train_fishing_model()