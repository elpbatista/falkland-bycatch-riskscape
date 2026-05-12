"""Compare joint Bayesian GMM species models with different component counts."""

from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_SWEEP_NAME = "bayesian_gmm_component_sweep_random12"
DEFAULT_METRICS_PATH = METRICS_DIR / "bayesian_gmm_component_comparison_random12.csv"
DEFAULT_TEST_FRACTION = 0.12
HOTSPOT_FRACTIONS = (0.10, 0.05, 0.01)


def add_hotspot_metrics(
    row: dict,
    y_true_log: pd.Series,
    pred_log: np.ndarray,
) -> None:
    """Add upper-tail and rank-based hotspot diagnostics."""
    y_true = np.expm1(y_true_log.to_numpy(dtype="float64"))
    pred = np.expm1(pred_log.astype("float64"))
    pred = np.maximum(pred, 0.0)

    for fraction in HOTSPOT_FRACTIONS:
        label = f"top_{int(fraction * 100)}"
        threshold = float(np.quantile(y_true, 1.0 - fraction))
        observed_hotspot = y_true >= threshold
        n_hotspot = int(observed_hotspot.sum())

        if n_hotspot == 0:
            row[f"{label}_observed_threshold"] = threshold
            row[f"{label}_observed_rows"] = 0
            row[f"{label}_mae"] = np.nan
            row[f"{label}_rmse"] = np.nan
            row[f"{label}_mean_pred"] = np.nan
            row[f"{label}_mean_obs"] = np.nan
            row[f"{label}_recall_at_k"] = np.nan
            row[f"{label}_capture_fraction"] = np.nan
            continue

        errors = pred[observed_hotspot] - y_true[observed_hotspot]
        ranked_pred = np.argsort(pred)[::-1][:n_hotspot]
        predicted_hotspot = np.zeros(len(pred), dtype=bool)
        predicted_hotspot[ranked_pred] = True
        hits = observed_hotspot & predicted_hotspot

        observed_sum = float(y_true[observed_hotspot].sum())
        captured_sum = float(y_true[hits].sum())

        row[f"{label}_observed_threshold"] = threshold
        row[f"{label}_observed_rows"] = n_hotspot
        row[f"{label}_mae"] = float(np.mean(np.abs(errors)))
        row[f"{label}_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        row[f"{label}_mean_pred"] = float(pred[observed_hotspot].mean())
        row[f"{label}_mean_obs"] = float(y_true[observed_hotspot].mean())
        row[f"{label}_pred_obs_ratio"] = float(
            pred[observed_hotspot].mean() / y_true[observed_hotspot].mean()
        )
        row[f"{label}_recall_at_k"] = float(hits.sum() / n_hotspot)
        row[f"{label}_capture_fraction"] = float(captured_sum / observed_sum)


def fit_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_components: int,
    sweep_name: str,
) -> dict:
    """Train and evaluate one component-count candidate."""
    x_train_num = train[FEATURES].to_numpy(dtype="float32")
    x_test_num = test[FEATURES].to_numpy(dtype="float32")

    y_train = train["_y"]
    y_test = test["_y"]

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_train_species = np.asarray(
        enc.fit_transform(train[["species"]]),
        dtype="float32",
    )
    x_test_species = np.asarray(
        enc.transform(test[["species"]]),
        dtype="float32",
    )

    x_train = np.concatenate((x_train_species, x_train_num), axis=1)
    x_test = np.concatenate((x_test_species, x_test_num), axis=1)

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
    add_hotspot_metrics(row, y_test, pred_log)
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
        "model_name": sweep_name,
        "model_type": "joint_species",
        "n_components": int(n_components),
    }

    model_path = (
        MODEL_DIR
        / sweep_name
        / f"components={n_components}"
        / "species_model_joint.joblib"
    )
    save_payload(payload, model_path)
    row["model_path"] = str(model_path)

    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Random holdout fraction for the component sweep.",
    )
    parser.add_argument(
        "--components",
        default=",".join(str(value) for value in COMPONENT_COUNTS),
        help="Comma-separated component counts to evaluate.",
    )
    parser.add_argument(
        "--sweep-name",
        default=DEFAULT_SWEEP_NAME,
        help="Model output folder name under data/modeling/models.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="CSV path for sweep metrics.",
    )

    return parser.parse_args()


def parse_components(value: str) -> list[int]:
    """Parse comma-separated component counts."""
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    component_counts = parse_components(args.components)
    df = prepare_species_table()
    df = sample_training_rows(df)
    train, test = split_random(df, test_fraction=args.test_fraction)

    rows = []
    for n_components in component_counts:
        print(f"Training Bayesian GMM with {n_components} components")
        row = fit_candidate(
            train,
            test,
            n_components,
            sweep_name=args.sweep_name,
        )
        row["test_fraction"] = float(args.test_fraction)
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

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.metrics_path, index=False)

    print(f"Saved comparison: {args.metrics_path}")
    print(out[["n_components", "rmse", "mae", "r2", "gmm_bic", "gmm_aic"]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
