"""Plot observed versus predicted species-use values for a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, SPECIES_TARGET
from riskscape.model.train import load_table, sample_training_rows, split_random


MODEL_NAMES = [
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
]
OUTPUT_ROOT = paths["plots"] / "species_use_diagnostics"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "species_use_diagnostics"


def load_test_predictions(model_name: str) -> pd.DataFrame:
    """Load the held-out rows and predict species use with a joint model."""
    payload_path = (
        paths["data"]
        / "modeling"
        / "models"
        / model_name
        / "species_model_joint.joblib"
    )
    payload = joblib.load(payload_path)

    df = load_table("species_training")
    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols).copy()
    df["_y"] = np.log1p(df[SPECIES_TARGET])
    df = sample_training_rows(df)

    _, test = split_random(df)

    encoder = payload["encoder"]
    model = payload["model"]
    feature_columns = list(payload["features"])

    x_species = encoder.transform(test[["species"]])
    x_features = test[feature_columns].to_numpy()
    x_test = np.hstack([x_species, x_features])

    pred_log = model.predict(x_test)
    observed = test[SPECIES_TARGET].to_numpy(dtype="float64")
    predicted = np.maximum(np.expm1(pred_log), 0.0)
    observed_log = np.log1p(observed)
    predicted_log = np.log1p(predicted)

    out = pd.DataFrame(
        {
            "species": test["species"].to_numpy(),
            "date": test["date"].to_numpy(),
            "observed_residence_index": observed,
            "predicted_residence_index": predicted,
            "residual": predicted - observed,
            "observed_log_transformed": observed_log,
            "predicted_log_transformed": predicted_log,
            "log_residual": predicted_log - observed_log,
        }
    )

    return out


def model_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Return regression metrics on the original residence-index scale."""
    observed = df["observed_residence_index"].to_numpy()
    predicted = df["predicted_residence_index"].to_numpy()
    mse = mean_squared_error(observed, predicted)

    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(observed, predicted)),
    }


def save_observed_vs_predicted(
    df: pd.DataFrame,
    model_name: str,
    out_file: Path,
) -> None:
    """Save a dense hexbin plot of observed versus predicted residence index."""
    metrics = model_metrics(df)
    observed = df["observed_residence_index"].to_numpy()
    predicted = df["predicted_residence_index"].to_numpy()
    observed_plot = np.log1p(observed)
    predicted_plot = np.log1p(predicted)
    max_value = max(float(observed_plot.max()), float(predicted_plot.max()))

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    hb = ax.hexbin(
        observed_plot,
        predicted_plot,
        gridsize=115,
        extent=(0, max_value, 0, max_value),
        mincnt=1,
        cmap="viridis",
        bins="log",
        linewidths=0.0,
    )
    ax.plot(
        [0, max_value],
        [0, max_value],
        color="#333333",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_xlabel("Observed log-transformed residence index", fontsize=8)
    ax.set_ylabel("Predicted log-transformed residence index", fontsize=8)
    ax.set_title(
        f"Observed vs Predicted Species Use — {model_name.replace('_', ' ').title()}",
        fontsize=10,
    )
    ax.tick_params(axis="both", labelsize=7)

    label = (
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f}\n"
        f"MAE = {metrics['mae']:.2f}"
    )
    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.88,
        },
    )

    cbar = fig.colorbar(hb, ax=ax, shrink=0.82)
    cbar.set_label("Held-out cells", fontsize=7)
    cbar.minorticks_off()
    cbar.ax.tick_params(which="major", length=2, labelsize=7)
    cbar.ax.tick_params(which="minor", length=0)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_residual_distribution(
    df: pd.DataFrame,
    model_name: str,
    out_file: Path,
) -> None:
    """Save a histogram of log-transformed residuals."""
    residuals = df["log_residual"].to_numpy()
    limit = float(np.nanquantile(np.abs(residuals), 0.995))

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.hist(
        residuals,
        bins=80,
        range=(-limit, limit),
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.25,
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)

    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Prediction residual on log-transformed scale", fontsize=8)
    ax.set_ylabel("Held-out cells", fontsize=8)
    ax.set_title(
        f"Residual Distribution — {model_name.replace('_', ' ').title()}",
        fontsize=10,
    )
    ax.tick_params(axis="both", labelsize=7)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot observed versus predicted species-use values.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help=(
            "Comma-separated model folder names under data/modeling/models, "
            "or 'tree_based' for the three tree-based species-use models."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated plot images.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated CSV exports.",
    )

    return parser.parse_args()


def parse_models(value: str) -> list[str]:
    """Parse model names from a command-line value."""
    if value == "tree_based":
        return MODEL_NAMES

    models = [model.strip() for model in value.split(",") if model.strip()]

    if not models:
        raise ValueError("At least one model name is required")

    return models


def save_model_outputs(
    model_name: str,
    output_root: Path,
    data_output_root: Path,
) -> None:
    """Save observed-versus-predicted diagnostics for one model."""
    df = load_test_predictions(model_name)
    safe_model = model_name.replace("/", "_")

    values_file = data_output_root / f"{safe_model}_observed_vs_predicted.csv"
    plot_file = output_root / f"{safe_model}_observed_vs_predicted.png"
    metrics_file = data_output_root / f"{safe_model}_observed_vs_predicted_metrics.csv"
    residual_file = output_root / f"{safe_model}_log_residual_distribution.png"

    values_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(values_file, index=False)
    pd.DataFrame([model_metrics(df)]).to_csv(metrics_file, index=False)
    save_observed_vs_predicted(df, model_name, plot_file)
    save_residual_distribution(df, model_name, residual_file)

    print("Saved:", values_file)
    print("Saved:", metrics_file)
    print("Saved:", plot_file)
    print("Saved:", residual_file)


def main() -> int:
    """Run the plotting workflow."""
    args = parse_args()
    model_names = parse_models(args.models)

    for model_name in model_names:
        save_model_outputs(model_name, args.output_root, args.data_output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
