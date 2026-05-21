"""Evaluate trained baseline models."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, FISHING_TARGET, SPECIES_TARGET
from riskscape.model.train import load_table, split_random


RANDOM_STATE = 42

MODEL_DIR = paths["data"] / "modeling" / "models"
EVAL_DIR = paths["data"] / "modeling" / "evaluation"


def metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n": int(len(y_true)),
    }


def save_predicted_vs_observed(
    y_true,
    y_pred,
    title: str,
    out_file: Path,
) -> None:
    """Save predicted vs observed scatter plot."""
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_log_space(
    y_true,
    pred_log,
    title: str,
    out_file: Path,
) -> None:
    """Save log-space predicted vs observed plot."""
    y_true_log = np.log1p(y_true)
    max_val = max(float(np.max(y_true_log)), float(np.max(pred_log)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_log, pred_log, s=4, alpha=0.25)
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)

    ax.set_xlabel("Observed log1p")
    ax.set_ylabel("Predicted log")
    ax.set_title(title)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    y_true,
    y_pred,
    title: str,
    out_file: Path,
) -> None:
    """Save residual plot."""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, residuals, s=4, alpha=0.25)
    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xlabel("Observed")
    ax.set_ylabel("Residual (pred - obs)")
    ax.set_title(title)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_density(
    y_true,
    y_pred,
    title: str,
    out_file: Path,
) -> None:
    """Save density scatter plot."""
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))

    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=50,
        mincnt=1,
    )

    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1)
    fig.colorbar(hb, ax=ax, label="Count")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_species_table() -> pd.DataFrame:
    """Load species table using same target preparation as training."""
    df = load_table("species_training")

    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols).copy()

    df[SPECIES_TARGET] = df[SPECIES_TARGET].clip(upper=20)
    df["_y"] = np.log1p(df[SPECIES_TARGET])

    return df


def split_time(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by date into train and test sets."""
    out = df.sort_values("date").reset_index(drop=True)
    split_index = int(len(out) * (1.0 - test_fraction))

    train = out.iloc[:split_index].copy()
    test = out.iloc[split_index:].copy()

    return train, test


def evaluate_single_species(species_name: str) -> dict:
    """Evaluate one single-species model."""
    df = prepare_species_table()
    df = df[df["species"] == species_name].copy()

    _, test = split_random(df)

    payload = joblib.load(
        MODEL_DIR / f"species_model_{species_name.lower()}.joblib"
    )

    model = payload["model"]

    y_true = test[SPECIES_TARGET].to_numpy()
    pred_log = model.predict(test[FEATURES])
    y_pred = np.expm1(pred_log)
    y_pred = np.maximum(y_pred, 0.0)

    result = metrics(y_true, y_pred)
    result["model"] = f"species_{species_name.lower()}"
    result["species"] = species_name

    prefix = EVAL_DIR / f"species_{species_name.lower()}"

    save_predicted_vs_observed(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{species_name}: predicted vs observed residence index",
        out_file=prefix.with_name(f"{prefix.name}_pred_vs_obs.png"),
    )

    plot_log_space(
        y_true=y_true,
        pred_log=pred_log,
        title=f"{species_name}: log-space predicted vs observed",
        out_file=prefix.with_name(f"{prefix.name}_log_space.png"),
    )

    plot_residuals(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{species_name}: residuals",
        out_file=prefix.with_name(f"{prefix.name}_residuals.png"),
    )

    plot_density(
        y_true=y_true,
        y_pred=y_pred,
        title=f"{species_name}: density scatter",
        out_file=prefix.with_name(f"{prefix.name}_density.png"),
    )

    return result


def evaluate_species_models() -> None:
    """Evaluate separate species models."""
    df = prepare_species_table()
    species_values = sorted(df["species"].dropna().unique().tolist())

    rows = []

    for species_name in species_values:
        rows.append(evaluate_single_species(species_name))

    out = pd.DataFrame(rows)
    out_file = EVAL_DIR / "species_separate_metrics.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)

    print(out)
    print("Saved:", out_file)


def evaluate_fishing_model() -> None:
    """Evaluate fishing model."""
    df = load_table("fishing_training")

    cols = [FISHING_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols).copy()
    df["_y"] = np.log1p(df[FISHING_TARGET])

    _, test = split_time(df)

    if len(test) > 1_000_000:
        test = test.sample(
            n=1_000_000,
            random_state=RANDOM_STATE,
        )

    payload = joblib.load(MODEL_DIR / "fishing_model.joblib")
    model = payload["model"]

    y_true = test[FISHING_TARGET].to_numpy()
    pred_log = model.predict(test[FEATURES])
    y_pred = np.expm1(pred_log)
    y_pred = np.maximum(y_pred, 0.0)

    result = metrics(y_true, y_pred)
    result["model"] = "fishing"

    out = pd.DataFrame([result])
    out_file = EVAL_DIR / "fishing_metrics.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)

    save_predicted_vs_observed(
        y_true=y_true,
        y_pred=y_pred,
        title="Fishing activity: predicted vs observed",
        out_file=EVAL_DIR / "fishing_pred_vs_obs.png",
    )

    plot_log_space(
        y_true=y_true,
        pred_log=pred_log,
        title="Fishing activity: log-space predicted vs observed",
        out_file=EVAL_DIR / "fishing_log_space.png",
    )

    plot_residuals(
        y_true=y_true,
        y_pred=y_pred,
        title="Fishing activity: residuals",
        out_file=EVAL_DIR / "fishing_residuals.png",
    )

    plot_density(
        y_true=y_true,
        y_pred=y_pred,
        title="Fishing activity: density scatter",
        out_file=EVAL_DIR / "fishing_density.png",
    )

    print(out)
    print("Saved:", out_file)


def evaluate_models() -> None:
    """Run model evaluation."""
    evaluate_species_models()
    # evaluate_fishing_model()
