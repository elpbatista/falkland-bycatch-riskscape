"""Layer 1b: Dynamic seascape classification (robust clustering with sampling)."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump

from riskscape.config import cfg, paths


def year_range():
    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year
    return range(start_year, end_year + 1)


def load_all_layer1():
    """Load and concatenate all Layer 1 data."""

    layer1_dir = Path(paths["layer1"])
    dfs = []

    for year in year_range():
        path = layer1_dir / f"year={year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing Layer 1 file: {path}")

        df = pd.read_parquet(
            path,
            columns=["date", "h3", "sst", "chl", "ssh"],
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def safe_log_chl(chl):
    """Log-transform CHL safely."""
    chl = chl.copy()
    chl[chl <= 0] = np.nan
    return np.log10(chl)


def fill_chl(X):
    """Fill NaN CHL with median (per batch)."""
    X = X.copy()

    chl = X[:, 1]

    if np.isnan(chl).any():
        median = np.nanmedian(chl)
        chl[np.isnan(chl)] = median

    X[:, 1] = chl
    return X


def prepare_features(df):
    """Prepare feature matrix and valid mask."""

    df = df.copy()

    df["chl"] = safe_log_chl(df["chl"].to_numpy(dtype=np.float32))

    X = df[["sst", "chl", "ssh"]].to_numpy(dtype=np.float32)

    # Only require SST and SSH
    valid_mask = (
        np.isfinite(X[:, 0]) &  # sst
        np.isfinite(X[:, 2])    # ssh
    )

    return df, X, valid_mask


def fit_model(X, k):
    """Fit scaler and clustering model."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(X_scaled)

    return scaler, model


def assign_regimes(df, labels):
    """Attach regime labels."""

    df = df.copy()

    regime = labels.astype("float32")
    regime[regime < 0] = np.nan

    df["regime_id"] = regime

    return df[["date", "h3", "regime_id"]]


def save_by_year(df):
    """Save Layer 1b outputs by year."""

    out_dir = Path(paths["data"]) / "layer1b"
    out_dir.mkdir(parents=True, exist_ok=True)

    df["year"] = pd.to_datetime(df["date"]).dt.year

    for year, group in df.groupby("year"):
        out_path = out_dir / f"year={year}.parquet"

        group = group.drop(columns="year")

        group.to_parquet(out_path, index=False)

        print(f"Saved: {out_path}")
        print(f"Rows: {len(group)}")


def save_artifacts(scaler, model, k):
    """Save scaler and clustering model."""

    model_dir = Path(paths["data"]) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = model_dir / "layer1b_scaler.joblib"
    model_path = model_dir / f"layer1b_k{k}.joblib"

    dump(scaler, scaler_path)
    dump(model, model_path)

    print(f"Saved scaler: {scaler_path}")
    print(f"Saved model: {model_path}")


def main():

    k = 7

    print("Loading Layer 1 data")
    df = load_all_layer1()

    df["date"] = pd.to_datetime(df["date"])

    print("Preparing features")
    df, X, valid_mask = prepare_features(df)

    valid_idx = np.where(valid_mask)[0]

    print(f"Valid rows: {len(valid_idx)} / {len(df)}")

    # --- SAMPLING ---
    sample_size = min(2_000_000, len(valid_idx))
    np.random.seed(42)

    sample_idx = np.random.choice(valid_idx, size=sample_size, replace=False)

    print(f"Training sample size: {sample_size}")

    # --- TRAINING DATA ---
    X_sample = X[sample_idx]

    # Keep rows with valid SST + SSH
    finite_mask_sample = (
        np.isfinite(X_sample[:, 0]) &
        np.isfinite(X_sample[:, 2])
    )

    X_sample = X_sample[finite_mask_sample]

    # Fill CHL before scaling
    X_sample = fill_chl(X_sample)

    print(f"Training after filter: {len(X_sample)}")

    # --- FIT MODEL ---
    print(f"Fitting clustering model (k={k})")
    scaler, model = fit_model(X_sample, k)

    print("Saving scaler and model")
    save_artifacts(scaler, model, k)

    # --- PREDICT FULL DATASET ---
    print("Predicting full dataset")

    X_valid = X[valid_mask]

    finite_mask_valid = (
        np.isfinite(X_valid[:, 0]) &
        np.isfinite(X_valid[:, 2])
    )

    labels_valid = np.full(len(X_valid), -1, dtype="int16")

    if finite_mask_valid.any():
        X_subset = X_valid[finite_mask_valid]

        # Fill CHL before scaling
        X_subset = fill_chl(X_subset)

        X_scaled = scaler.transform(X_subset)
        labels_valid[finite_mask_valid] = model.predict(X_scaled)

    labels = np.full(len(df), -1, dtype="int16")
    labels[valid_mask] = labels_valid

    # --- ASSIGN ---
    print("Assigning regimes")
    df_regimes = assign_regimes(df, labels)

    # --- SAVE ---
    print("Saving outputs")
    save_by_year(df_regimes)

    print("Done")


if __name__ == "__main__":
    main()