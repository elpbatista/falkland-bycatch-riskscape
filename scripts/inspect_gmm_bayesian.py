"""Inspect GMM/Bayesian ecological regimes."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import FEATURES


MODEL_NAME = "bayesian_gmm"
SPECIES = "BBAL"

MODEL_PATH = (
    paths["data"]
    / "modeling"
    / "models"
    / MODEL_NAME
    / f"species_model_{SPECIES.lower()}.joblib"
)

OUT_PATH = (
    paths["data"]
    / "modeling"
    / "metrics"
    / f"{MODEL_NAME}_{SPECIES.lower()}_gmm_components.csv"
)


def main() -> int:
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]

    means_scaled = model.gmm.means_
    means_raw = model.scaler.inverse_transform(means_scaled)

    std_scaled = np.sqrt(
        np.array([np.diag(cov) for cov in model.gmm.covariances_])
    )

    std_raw = std_scaled * model.scaler.scale_

    means = pd.DataFrame(means_raw, columns=FEATURES)
    std = pd.DataFrame(std_raw, columns=[f"{col}_std" for col in FEATURES])

    out = pd.concat([means, std], axis=1)
    out.insert(0, "component", range(len(out)))
    out.insert(1, "weight", model.gmm.weights_)

    out = out.sort_values("weight", ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(out)
    print(f"Saved: {OUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())