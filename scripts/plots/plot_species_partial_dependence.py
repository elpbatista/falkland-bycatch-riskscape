"""Plot manual partial dependence for the selected species-use model."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, SPECIES_TARGET
from riskscape.model.train import load_table, sample_training_rows


MODEL_NAME = "extra_trees"
OUTPUT_ROOT = paths["plots"] / "species_use_diagnostics"
DATA_OUTPUT_ROOT = paths["data"] / "plot_exports" / "species_use_diagnostics"
SAMPLE_SIZE = 8_000
GRID_SIZE = 60
RANDOM_STATE = 42


@dataclass(frozen=True)
class PredictorSpec:
    """Display and transform settings for one partial-dependence predictor."""

    column: str
    label: str
    xlabel: str
    transform: str | None = None


PREDICTOR_SPECS = [
    PredictorSpec("depth_m", "Bathymetry", "Depth (m)"),
    PredictorSpec("dist_coast_m", "Distance to coast", "Distance to coast (km)", "m_to_km"),
    PredictorSpec("ssh_anom", "SSH anomaly", "SSH anomaly (m)"),
    PredictorSpec("ssh", "SSH", "SSH (m)"),
    PredictorSpec("sst", "SST", "SST (C)", "kelvin_to_c"),
    PredictorSpec("chl_log_grad", "CHL gradient", "CHL local gradient (log units)"),
]


def display_values(values: np.ndarray, transform: str | None) -> np.ndarray:
    """Return values transformed for axis display."""
    if transform == "kelvin_to_c":
        return values - 273.15

    if transform == "m_to_km":
        return values / 1_000.0

    return values


def load_model_payload(model_name: str) -> dict:
    """Load a joint species-use model payload."""
    path = (
        paths["data"]
        / "modeling"
        / "models"
        / model_name
        / "species_model_joint.joblib"
    )

    return joblib.load(path)


def load_sample(sample_size: int) -> pd.DataFrame:
    """Load a reproducible sample of the species-use training rows."""
    df = load_table("species_training")
    cols = ["species", SPECIES_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols).copy()
    df = sample_training_rows(df)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)


def predict_log(payload: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict log-transformed species use from a joint model payload."""
    encoder = payload["encoder"]
    model = payload["model"]
    feature_columns = list(payload["features"])

    x_species = encoder.transform(df[["species"]])
    x_features = df[feature_columns].to_numpy()
    x = np.hstack([x_species, x_features])

    return model.predict(x)


def partial_dependence(
    payload: dict,
    df: pd.DataFrame,
    spec: PredictorSpec,
    grid_size: int,
) -> pd.DataFrame:
    """Compute manual partial dependence for one predictor."""
    values = df[spec.column].to_numpy()
    grid = np.linspace(
        float(np.quantile(values, 0.02)),
        float(np.quantile(values, 0.98)),
        grid_size,
    )
    pd_values: list[float] = []

    for value in grid:
        tmp = df.copy()
        tmp[spec.column] = value
        pred = predict_log(payload, tmp)
        pd_values.append(float(np.mean(pred)))

    return pd.DataFrame(
        {
            "predictor": spec.column,
            "x": display_values(grid, spec.transform),
            "partial_dependence": pd_values,
        }
    )


def save_partial_dependence_plot(
    curves: dict[PredictorSpec, pd.DataFrame],
    out_file: Path,
) -> None:
    """Save a multi-panel partial-dependence figure."""
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2))
    axes_flat = axes.ravel()

    for ax, (spec, curve) in zip(axes_flat, curves.items(), strict=True):
        ax.plot(
            curve["x"],
            curve["partial_dependence"],
            color="#4c78a8",
            linewidth=1.7,
        )
        ax.set_title(spec.label, fontsize=9)
        ax.set_xlabel(spec.xlabel, fontsize=7)
        ax.set_ylabel("Mean predicted log-use", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(color="#dddddd", linewidth=0.45)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Selected Species-Use Model — Partial Dependence",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot partial dependence for selected species-use predictors.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model folder name under data/modeling/models.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Number of rows to sample for partial dependence averaging.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Number of values in each predictor grid.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated partial-dependence figures.",
    )
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=DATA_OUTPUT_ROOT,
        help="Directory for generated partial-dependence CSV exports.",
    )

    return parser.parse_args()


def main() -> int:
    """Run partial-dependence plotting."""
    args = parse_args()
    payload = load_model_payload(args.model)
    df = load_sample(args.sample_size)

    curves = {
        spec: partial_dependence(payload, df, spec, args.grid_size)
        for spec in PREDICTOR_SPECS
    }

    out = pd.concat(curves.values(), ignore_index=True)
    out_csv = args.data_output_root / f"{args.model}_partial_dependence.csv"
    out_png = args.output_root / f"{args.model}_partial_dependence.png"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    save_partial_dependence_plot(curves, out_png)

    print("Saved:", out_csv)
    print("Saved:", out_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
