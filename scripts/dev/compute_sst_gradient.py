import numpy as np
import pandas as pd
from pathlib import Path

from riskscape.config import paths


def compute_gradient_year(year):

    print(f"\nProcessing year {year}")

    layer1_path = Path(paths["layer1"]) / f"year={year}.parquet"
    neighbor_path = Path(paths["processed"]) / "h3_neighbor_index.parquet"

    df = pd.read_parquet(layer1_path)
    neighbors = pd.read_parquet(neighbor_path)

    # Ensure consistent ordering
    df = df.sort_values(["date", "h3"]).reset_index(drop=True)

    # Build index mapping
    h3_sorted = sorted(df["h3"].unique())
    h3_to_idx = {h: i for i, h in enumerate(h3_sorted)}

    df["idx"] = df["h3"].map(h3_to_idx)

    # Pivot to matrix: time × cell
    mat = (
        df.pivot(index="date", columns="idx", values="sst")
        .sort_index(axis=1)
        .values
    )

    # Convert Kelvin → Celsius
    mat = mat - 273.15

    neighbor_array = neighbors.sort_values("idx")[
        ["n1", "n2", "n3", "n4", "n5", "n6"]
    ].values

    gradients = np.zeros_like(mat)

    for t in range(mat.shape[0]):

        X = mat[t]

        diffs = []

        for k in range(6):
            n_idx = neighbor_array[:, k]

            valid = n_idx >= 0
            diff = np.zeros_like(X)

            diff[valid] = X[valid] - X[n_idx[valid]]
            diffs.append(diff ** 2)

        diffs = np.stack(diffs, axis=0)

        # mean over neighbors ignoring invalid
        counts = (neighbor_array >= 0).sum(axis=1)
        mean_sq = diffs.sum(axis=0) / counts

        gradients[t] = np.sqrt(mean_sq)

    # Reconstruct DataFrame
    grad_df = pd.DataFrame(
        gradients,
        index=df["date"].unique(),
        columns=h3_sorted
    )

    grad_df = (
        grad_df.stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "h3", 0: "sst_grad"})
    )

    output_path = Path(paths["processed"]) / f"sst_gradient_{year}.parquet"
    grad_df.to_parquet(output_path, index=False)

    print("Saved:", output_path)


if __name__ == "__main__":
    compute_gradient_year(2014)