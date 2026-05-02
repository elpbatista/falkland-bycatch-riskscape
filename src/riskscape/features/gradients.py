"""Spatial gradients on H3 feature tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskscape.config import paths


VARIABLES = ["sst", "chl_log", "ssh"]


def load_neighbor_index() -> pd.DataFrame:
    """Load H3 neighbor index table."""
    path = paths["data"] / "processed" / "h3_neighbor_index.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing neighbor index: {path}")

    return pd.read_parquet(path)


def load_h3_order() -> pd.Series:
    """Load H3 order matching neighbor index."""
    path = paths["data"] / "processed" / "h3_neighbors.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing neighbors table: {path}")

    df = pd.read_parquet(path)

    if "h3" not in df.columns:
        raise KeyError("Missing h3 column in h3_neighbors.parquet")

    return df["h3"].astype("uint64")


def build_neighbor_array(index_df: pd.DataFrame) -> np.ndarray:
    """Build neighbor index array."""
    cols = ["n1", "n2", "n3", "n4", "n5", "n6"]

    missing = [col for col in cols if col not in index_df.columns]
    if missing:
        raise KeyError(f"Missing neighbor columns: {missing}")

    return index_df[cols].to_numpy(dtype="int64")


def compute_group_gradients(
    group: pd.DataFrame,
    h3_order: pd.Series,
    neighbor_idx: np.ndarray,
    variables: list[str],
) -> pd.DataFrame:
    """Compute gradients for one date."""
    out = group[["h3", "date"]].copy()

    ordered = group.set_index("h3").reindex(h3_order)

    valid_neighbors = neighbor_idx >= 0
    safe_idx = np.where(valid_neighbors, neighbor_idx, 0)

    for var in variables:
        values = ordered[var].to_numpy(dtype="float64")

        center = values[:, None]
        neigh = values[safe_idx]
        neigh = np.where(valid_neighbors, neigh, np.nan)

        diff2 = (center - neigh) ** 2

        valid = np.isfinite(diff2)
        counts = valid.sum(axis=1)

        sums = np.nansum(diff2, axis=1)

        grad = np.full(len(values), np.nan, dtype="float32")
        ok = counts > 0

        grad[ok] = np.sqrt(sums[ok] / counts[ok]).astype("float32")

        grad_df = pd.DataFrame(
            {
                "h3": h3_order.to_numpy(),
                f"{var}_grad": grad,
            }
        )

        out = out.merge(grad_df, on="h3", how="left")

    return out


def add_gradients(
    df: pd.DataFrame,
    h3_order: pd.Series,
    neighbor_idx: np.ndarray,
    variables: list[str],
) -> pd.DataFrame:
    """Add H3 neighbor gradients."""
    required = {"h3", "date"} | set(variables)
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["h3"] = df["h3"].astype("uint64")

    frames = []

    for _, group in df.groupby("date", sort=False):
        frames.append(
            compute_group_gradients(
                group=group,
                h3_order=h3_order,
                neighbor_idx=neighbor_idx,
                variables=variables,
            )
        )

    gradients = pd.concat(frames, ignore_index=True)

    drop_cols = [f"{var}_grad" for var in variables]
    existing = [col for col in drop_cols if col in df.columns]

    if existing:
        df = df.drop(columns=existing)

    return df.merge(gradients, on=["h3", "date"], how="left")


def process_environmental_gradients() -> None:
    """Add spatial gradients to environmental partitions."""
    root = paths["data"] / "features" / "environmental"
    parts = sorted(root.glob("year=*/part.parquet"))

    if not parts:
        raise FileNotFoundError(f"No environmental partitions found: {root}")

    h3_order = load_h3_order()
    index_df = load_neighbor_index()
    neighbor_idx = build_neighbor_array(index_df)

    if len(h3_order) != len(index_df):
        raise ValueError("h3_neighbors and h3_neighbor_index row counts differ")

    for path in parts:
        df = pd.read_parquet(path)

        df = add_gradients(
            df=df,
            h3_order=h3_order,
            neighbor_idx=neighbor_idx,
            variables=VARIABLES,
        )

        df.to_parquet(path, index=False, compression="zstd")

        print(f"Updated gradients: {path}")