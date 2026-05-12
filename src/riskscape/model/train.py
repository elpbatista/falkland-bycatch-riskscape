"""Train species-use models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, SPECIES_TARGET
from riskscape.utils.dates import normalize_date_column


RANDOM_STATE = 42

MODEL_NAMES = [
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    # "gmm",
    "bayesian_gmm",
]

MAX_ZERO_ROWS = 250_000
MAX_POSITIVE_ROWS = None

MODEL_DIR = paths["data"] / "modeling" / "models"
METRICS_DIR = paths["data"] / "modeling" / "metrics"
ACTIVE_EXTRA_TREES_MODEL_NAME = "extra_trees_kmeans_k15_blockcv"

GMM_COMPONENTS = 30
GMM_REG_COVAR = 1e-6


HIST_GRADIENT_BOOSTING_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.1,
}

BAYESIAN_PRIOR_PARAMS = {
    "max_iter": 150,
    "max_leaf_nodes": 15,
}


def build_hist_gradient_boosting(
    random_state: int = RANDOM_STATE,
    **overrides,
) -> HistGradientBoostingRegressor:
    """Build a histogram gradient boosting regressor."""
    params = {**HIST_GRADIENT_BOOSTING_PARAMS, **overrides}

    return HistGradientBoostingRegressor(
        **params,
        random_state=random_state,
    )


class GMMUseRegressor(BaseEstimator, RegressorMixin):
    """Estimate use from environmental likelihood."""

    def __init__(
        self,
        n_components: int = GMM_COMPONENTS,
        random_state: int = RANDOM_STATE,
        reg_covar: float = GMM_REG_COVAR,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=reg_covar,
            random_state=random_state,
        )
        self.min_log_density = 0.0
        self.max_log_density = 1.0
        self.max_target = 1.0

    def fit(self, x, y, sample_weight=None):
        """Fit density model on positive-use rows."""
        y_values = np.asarray(y)
        positive = y_values > 0

        if positive.sum() < self.n_components:
            raise ValueError("Not enough positive rows to fit GMM")

        x_positive = np.asarray(x)[positive]
        y_positive = y_values[positive]

        x_scaled = self.scaler.fit_transform(x_positive)
        self.gmm.fit(x_scaled)

        log_density = self.gmm.score_samples(x_scaled)
        self.min_log_density = float(np.quantile(log_density, 0.01))
        self.max_log_density = float(np.quantile(log_density, 0.99))
        self.max_target = float(np.quantile(y_positive, 0.99))

        if self.max_log_density <= self.min_log_density:
            self.max_log_density = self.min_log_density + 1.0

        return self

    def predict(self, x):
        """Predict log-space use from likelihood."""
        x_scaled = self.scaler.transform(x)
        log_density = self.gmm.score_samples(x_scaled)

        score = (
            (log_density - self.min_log_density)
            / (self.max_log_density - self.min_log_density)
        )
        score = np.clip(score, 0.0, 1.0)

        return score * self.max_target

    def predict_log_density(self, x):
        """Return raw log-density for presence modeling."""
        x_scaled = self.scaler.transform(x)
        return self.gmm.score_samples(x_scaled)


class BayesianGMMUseRegressor(GMMUseRegressor):
    """Estimate use from likelihood and a seasonal prior."""

    def __init__(
        self,
        n_components: int = GMM_COMPONENTS,
        random_state: int = RANDOM_STATE,
        reg_covar: float = GMM_REG_COVAR,
    ) -> None:
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            reg_covar=reg_covar,
        )
        self.prior_model = build_hist_gradient_boosting(
            random_state=random_state,
            **BAYESIAN_PRIOR_PARAMS,
        )

    def fit(self, x, y, sample_weight=None):
        """Fit likelihood and prior models."""
        super().fit(x, y, sample_weight=sample_weight)
        self.prior_model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        """Predict log-space use from likelihood and prior."""
        likelihood = super().predict(x)
        prior = self.prior_model.predict(x)
        prior = np.maximum(prior, 0.0)

        combined = 0.5 * likelihood + 0.5 * prior

        return combined

    def predict_log_density(self, x):
        """Return raw log-density for presence modeling."""
        x_scaled = self.scaler.transform(x)
        return self.gmm.score_samples(x_scaled)


def load_table(name: str) -> pd.DataFrame:
    """Load partitioned modeling table."""
    root = paths["data"] / "modeling" / name
    frames = []

    for path in sorted(root.glob("year=*/part.parquet")):
        frames.append(normalize_date_column(pd.read_parquet(path)))

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


# def nonzero_training_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def build_random_forest() -> RandomForestRegressor:
    """Build random forest regressor."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_extra_trees() -> ExtraTreesRegressor:
    """Build extra trees regressor."""
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gmm() -> GMMUseRegressor:
    """Build GMM use regressor."""
    return GMMUseRegressor()


def build_bayesian_gmm() -> BayesianGMMUseRegressor:
    """Build Bayesian GMM use regressor."""
    return BayesianGMMUseRegressor()


MODEL_BUILDERS = {
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "gmm": build_gmm,
    "bayesian_gmm": build_bayesian_gmm,
}


def build_model(model_name: str):
    """Build model by name."""
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

    return builder()


def sample_weights(y_train: pd.Series) -> np.ndarray:
    """Return sample weights from raw target scale."""
    raw_target = np.expm1(y_train)
    # weights = 1.0 + raw_target
    # weights = 1.0 + 2.0 * raw_target
    weights = 1.0 + raw_target ** 0.75
    # weights = 1.0 + raw_target ** 1.75
    # weights = 1 + 5 * (raw_target > 0)

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

    model_dir = MODEL_DIR / model_name
    save_payload(
        payload,
        model_dir / "species_model_joint.joblib",
    )

    print("Joint species model:", model_name)
    print(m)

    return m


def train_production_extra_trees_model() -> dict:
    """Train the active Extra Trees joint model on all balanced rows."""
    df = sample_training_rows(prepare_species_table())

    x_num = df[FEATURES].to_numpy()
    y = df["_y"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_species = enc.fit_transform(df[["species"]])
    x = np.hstack([x_species, x_num])

    model = build_extra_trees()
    model.fit(
        x,
        y,
        sample_weight=sample_weights(y),
    )

    pred_log = model.predict(x)
    m = evaluate_predictions(y, pred_log)
    log_m = metrics(y, pred_log)
    m.update(
        {
            "r2_log": log_m["r2"],
            "rmse_log": log_m["rmse"],
            "mae_log": log_m["mae"],
            "model": ACTIVE_EXTRA_TREES_MODEL_NAME,
            "base_model": "extra_trees",
            "model_type": "joint_species",
            "species": "all",
            "train_rows": int(len(df)),
            "test_rows": 0,
            "validation": "production_all_balanced_rows",
        }
    )

    payload = {
        "model": model,
        "encoder": enc,
        "features": FEATURES,
        "target": SPECIES_TARGET,
        "log_target": True,
        "model_name": ACTIVE_EXTRA_TREES_MODEL_NAME,
        "base_model": "extra_trees",
        "model_type": "joint_species",
        "species": "all",
        "validation": "production_all_balanced_rows",
    }

    model_dir = MODEL_DIR / ACTIVE_EXTRA_TREES_MODEL_NAME
    save_payload(
        payload,
        model_dir / "species_model_joint.joblib",
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"species_model_{ACTIVE_EXTRA_TREES_MODEL_NAME}_production_metrics.csv"
    pd.DataFrame([m]).to_csv(metrics_path, index=False)

    print("Production joint species model:", ACTIVE_EXTRA_TREES_MODEL_NAME)
    print(m)
    print("Metrics saved:", metrics_path)

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

    model_dir = MODEL_DIR / model_name
    save_payload(
        payload,
        model_dir / f"species_model_{safe_name}.joblib",
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

    rows = []

    for model_name in MODEL_NAMES:
        rows.append(train_joint_species_model(df, model_name))

    save_metrics(rows)
