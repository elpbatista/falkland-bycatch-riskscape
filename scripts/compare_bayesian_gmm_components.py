"""Compare joint Bayesian GMM species models with different component counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from riskscape.model.dataset import FEATURES, SPECIES_TARGET
from riskscape.model.train import (
    GMM_REG_COVAR,
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    BayesianGMMUseRegressor,
    evaluate_predictions,
    prepare_species_table,
    sample_training_rows,
    sample_weights,
    save_payload,
    split_random,
)


COMPONENT_COUNTS = [2, 4, 6, 8, 10, 12]
MODEL_NAME = "bayesian_gmm_component_sweep"
METRICS_PATH = METRICS_DIR / "bayesian_gmm_component_comparison.csv"


def fit_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int,
) -> dict:
    """Train and evaluate one component-count candidate."""
    x_train_num = train[FEATURES].to_numpy()
    x_test_num = test[FEATURES].to_numpy()

    y_train = train["_y"]
    y_test = test["_y"]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_train_species = enc.fit_transform(train[["species"]])
    x_test_species = enc.transform(test[["species"]])

    x_train = np.hstack([x_train_species, x_train_num])
    x_test = np.hstack([x_test_species, x_test_num])

    model = BayesianGMMUseRegressor(
        n_components=n_components,
        random_state=RANDOM_STATE,
        reg_covar=GMM_REG_COVAR,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights(y_train),
    )

    pred_log = model.predict(x_test)
    row = evaluate_predictions(y_test, pred_log)
    row["model"] = "bayesian_gmm"
    row["model_type"] = "joint_species"
    row["species"] = "all"
    row["n_components"] = int(n_components)
    row["train_rows"] = int(len(train))
    row["test_rows"] = int(len(test))
    row["positive_train_rows"] = int((train[SPECIES_TARGET] > 0).sum())

    positive = train[SPECIES_TARGET].to_numpy() > 0
    x_positive_scaled = model.scaler.transform(x_train[positive])
    row["gmm_bic"] = float(model.gmm.bic(x_positive_scaled))
    row["gmm_aic"] = float(model.gmm.aic(x_positive_scaled))
    row["gmm_mean_log_likelihood"] = float(model.gmm.score(x_positive_scaled))

    labels = model.gmm.predict(x_positive_scaled)
    counts = np.bincount(labels, minlength=n_components)
    for component, count in enumerate(counts):
        row[f"component_{component}_positive_rows"] = int(count)
        row[f"component_{component}_positive_fraction"] = float(count / positive.sum())

    payload = {
        "model": model,
        "encoder": enc,
        "features": FEATURES,
        "target": SPECIES_TARGET,
        "log_target": True,
        "model_name": MODEL_NAME,
        "model_type": "joint_species",
        "n_components": int(n_components),
    }

    model_path = (
        MODEL_DIR
        / MODEL_NAME
        / f"components={n_components}"
        / "species_model_joint.joblib"
    )
    save_payload(payload, model_path)
    row["model_path"] = str(model_path)

    return row


def main() -> int:
    df = prepare_species_table()
    df = sample_training_rows(df)
    train, test = split_random(df)

    rows = []
    for n_components in COMPONENT_COUNTS:
        print(f"Training Bayesian GMM with {n_components} components")
        row = fit_candidate(train, test, n_components)
        rows.append(row)
        print(
            {
                "n_components": row["n_components"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "gmm_bic": row["gmm_bic"],
                "gmm_aic": row["gmm_aic"],
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["rmse", "mae", "gmm_bic"]).reset_index(drop=True)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRICS_PATH, index=False)

    print(f"Saved comparison: {METRICS_PATH}")
    print(out[["n_components", "rmse", "mae", "r2", "gmm_bic", "gmm_aic"]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
