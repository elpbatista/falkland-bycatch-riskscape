"""Inspect model feature importance."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import pandas as pd

from riskscape.config import paths
from riskscape.model.dataset import FEATURES


MODEL_DIR = paths["data"] / "modeling" / "models"
ACTIVE_SPECIES_MODEL = "extra_trees_som_hierarchical_k30_5fold_blockcv"


def species_importance() -> pd.DataFrame:
    """Return species Extra Trees feature importance."""
    payload = joblib.load(
        MODEL_DIR / ACTIVE_SPECIES_MODEL / "species_model_joint.joblib"
    )

    model = payload["model"]
    encoder = payload["encoder"]

    species_names = encoder.get_feature_names_out(["species"]).tolist()
    names = species_names + FEATURES

    importance = pd.DataFrame(
        {
            "feature": names,
            "importance": model.feature_importances_,
        }
    )

    return importance.sort_values("importance", ascending=False)


def main() -> int:
    """Run inspection."""
    species = species_importance()

    print("Species feature importance:")
    print(species.to_string(index=False))

    species_file = MODEL_DIR / "species_feature_importance.csv"
    species.to_csv(species_file, index=False)

    print()
    print("Saved:", species_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
