"""Feature correlation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = PROJECT_ROOT / "outputs"


def load_partitioned_table(name: str) -> pd.DataFrame:
    root = paths["data"] / "features" / name
    frames = []

    for year_dir in sorted(root.glob("year=*")):
        path = year_dir / "part.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No data: {name}")

    return pd.concat(frames, ignore_index=True)


def load_static_table() -> pd.DataFrame:
    path = paths["data"] / "features" / "static" / "static.parquet"
    return pd.read_parquet(path)


def load_table(name: str) -> pd.DataFrame:
    if name == "static":
        return load_static_table()
    return load_partitioned_table(name)


def build_frame(cfg: dict) -> pd.DataFrame:
    tables = cfg["source_tables"]
    keys = cfg.get("join_keys", [])

    df = load_table(tables[0])

    for table in tables[1:]:
        other = load_table(table)
        df = df.merge(other, on=keys, how="inner")

    return df


def compute(df: pd.DataFrame, features: list[str], method: str) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    return df[features].dropna().corr(method=method)


def save_matrix(corr: pd.DataFrame, run_name: str) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / f"{run_name}_correlation_matrix.csv"
    corr.to_csv(out)


def plot_heatmap(corr: pd.DataFrame, run_name: str) -> None:
    OUTPUTS.mkdir(exist_ok=True)

    fig, ax = plt.subplots()

    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    out = OUTPUTS / f"{run_name}_correlation_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_correlation_analysis(run_name: str, cfg: dict) -> None:
    df = build_frame(cfg)

    corr = compute(
        df,
        features=cfg["features"],
        method=cfg.get("method", "spearman"),
    )

    save_matrix(corr, run_name)

    if cfg.get("plot", True):
        plot_heatmap(corr, run_name)