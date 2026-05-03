"""Inspect model feature importance."""

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, FISHING_TARGET, SPECIES_TARGET
from riskscape.model.train import load_table, split_time


RANDOM_STATE = 42
MAX_PERMUTATION_ROWS = 100_000
MODEL_DIR = paths["data"] / "modeling" / "models"


def species_importance() -> pd.DataFrame:
    """Return species Random Forest feature importance."""
    payload = joblib.load(MODEL_DIR / "species_model.joblib")

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


def fishing_importance() -> pd.DataFrame:
    """Return fishing permutation feature importance."""
    payload = joblib.load(MODEL_DIR / "fishing_model.joblib")
    model = payload["model"]

    df = load_table("fishing_training")

    cols = [FISHING_TARGET] + FEATURES
    df = df[cols + ["date"]].dropna(subset=cols)

    _, test = split_time(df)

    if len(test) > MAX_PERMUTATION_ROWS:
        test = test.sample(
            n=MAX_PERMUTATION_ROWS,
            random_state=RANDOM_STATE,
        )

    x_test = test[FEATURES]
    y_test = test[FISHING_TARGET]

    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance.sort_values("importance_mean", ascending=False)


def main() -> int:
    """Run inspection."""
    species = species_importance()
    fishing = fishing_importance()

    print("Species feature importance:")
    print(species.to_string(index=False))

    species_file = MODEL_DIR / "species_feature_importance.csv"
    species.to_csv(species_file, index=False)

    print()
    print("Fishing feature importance:")
    print(fishing.to_string(index=False))

    fishing_file = MODEL_DIR / "fishing_feature_importance.csv"
    fishing.to_csv(fishing_file, index=False)

    print()
    print("Saved:", species_file)
    print("Saved:", fishing_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())