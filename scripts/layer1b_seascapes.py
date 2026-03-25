"""Layer 1b: train on valid CHL, predict everywhere."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from riskscape.config import cfg, paths


def year_range():
    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def load_all_layer1():
    """Load all Layer 1 yearly files."""

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
    """Log-transform CHL, masking non-positive values."""

    chl = chl.copy()
    chl[chl <= 0] = np.nan
    return np.log10(chl)


def prepare_features(df):
    """Build feature matrix."""

    df = df.copy()
    df["chl"] = safe_log_chl(df["chl"].to_numpy(dtype=np.float32))
    x = df[["sst", "chl", "ssh"]].to_numpy(dtype=np.float32)
    return df, x


def fill_chl(x):
    """Fill missing CHL with the batch median."""

    x = x.copy()
    chl = x[:, 1]

    if np.isnan(chl).any():
        median = np.nanmedian(chl)
        chl[np.isnan(chl)] = median

    x[:, 1] = chl
    return x


def fit_model(x, k):
    """Fit scaler and clustering model."""

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    model.fit(x_scaled)

    return scaler, model


def assign_regimes(df, labels):
    """Attach regime labels to the table."""

    df = df.copy()
    regime = labels.astype("float32")
    regime[regime < 0] = np.nan
    df["regime_id"] = regime
    return df[["date", "h3", "regime_id"]]


def save_by_year(df):
    """Write Layer 1b outputs by year."""

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

    dump(scaler, model_dir / "layer1b_scaler.joblib")
    dump(model, model_dir / f"layer1b_k{k}.joblib")


def main():
    k = 7

    print("Loading Layer 1 data")
    df = load_all_layer1()
    df["date"] = pd.to_datetime(df["date"])

    print("Preparing features")
    df, x = prepare_features(df)

    # Train only where SST, CHL, and SSH are all available.
    train_mask = (
        np.isfinite(x[:, 0]) &
        np.isfinite(x[:, 1]) &
        np.isfinite(x[:, 2])
    )

    train_idx = np.where(train_mask)[0]
    print(f"Training rows: {len(train_idx)} / {len(df)}")

    sample_size = min(2_000_000, len(train_idx))
    np.random.seed(42)

    sample_idx = np.random.choice(train_idx, size=sample_size, replace=False)
    x_sample = x[sample_idx]

    print(f"Training sample size: {len(x_sample)}")

    print(f"Fitting clustering model (k={k})")
    scaler, model = fit_model(x_sample, k)

    save_artifacts(scaler, model, k)

    # Predict everywhere SST and SSH exist, imputing CHL only here.
    pred_mask = (
        np.isfinite(x[:, 0]) &
        np.isfinite(x[:, 2])
    )

    pred_idx = np.where(pred_mask)[0]
    print(f"Prediction rows: {len(pred_idx)} / {len(df)}")

    x_pred = fill_chl(x[pred_idx])
    x_scaled = scaler.transform(x_pred)
    labels_pred = model.predict(x_scaled)

    labels = np.full(len(df), -1, dtype="int16")
    labels[pred_idx] = labels_pred

    print("Assigning regimes")
    df_regimes = assign_regimes(df, labels)

    print("Saving outputs")
    save_by_year(df_regimes)

    print("Done")


if __name__ == "__main__":
    main()