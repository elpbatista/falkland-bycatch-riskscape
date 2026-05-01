"""Response vs predictor relationships."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = PROJECT_ROOT / "outputs"


def load_partitioned_table(table_name: str) -> pd.DataFrame:
    root = paths["data"] / "features" / table_name
    frames = []

    for year_dir in sorted(root.glob("year=*")):
        path = year_dir / "part.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No data for table: {table_name}")

    return pd.concat(frames, ignore_index=True)


def load_static_table() -> pd.DataFrame:
    path = paths["data"] / "features" / "static" / "static.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    return pd.read_parquet(path)


def load_feature_table(name: str) -> pd.DataFrame:
    if name == "static":
        return load_static_table()
    return load_partitioned_table(name)


def build_frame(cfg: dict) -> pd.DataFrame:
    df = load_feature_table(cfg["response_table"])

    if "response_filter" in cfg:
        for col, val in cfg["response_filter"].items():
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    df[cfg["response_name"]] = df.eval(cfg["response_expr"])

    keys = cfg["join_keys"]
    groups = cfg.get("groups", [])
    predictors = cfg["predictors"]

    keep = keys + groups + [cfg["response_name"]]
    df = df[keep].copy()

    for table in cfg["predictor_tables"]:
        other = load_feature_table(table)
        df = df.merge(other, on=keys, how="left")

    required = [cfg["response_name"]] + predictors
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


def binned_summary(
    df: pd.DataFrame,
    response: str,
    predictors: list[str],
    groups: list[str],
    bins: int,
) -> pd.DataFrame:
    rows = []

    for predictor in predictors:
        tmp = df[[predictor, response] + groups].dropna().copy()

        if tmp.empty:
            continue

        tmp["bin"] = pd.qcut(tmp[predictor], q=bins, duplicates="drop")

        agg = (
            tmp.groupby(groups + ["bin"], observed=True)
            .agg(
                predictor_mean=(predictor, "mean"),
                response_mean=(response, "mean"),
                response_count=(response, "count"),
            )
            .reset_index()
        )

        agg["predictor"] = predictor
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot(summary: pd.DataFrame, run_name: str, groups: list[str]) -> None:
    OUTPUTS.mkdir(exist_ok=True)

    for predictor in summary["predictor"].unique():
        df = summary[summary["predictor"] == predictor]

        if groups:
            grouped = df.groupby(groups)
        else:
            grouped = [("all", df)]

        for gname, gdf in grouped:
            if isinstance(gname, tuple):
                label = "_".join(str(x) for x in gname)
            else:
                label = str(gname)

            sizes = 20 + 2 * gdf["response_count"]

            fig, ax = plt.subplots()

            ax.plot(gdf["predictor_mean"], gdf["response_mean"])
            ax.scatter(
                gdf["predictor_mean"],
                gdf["response_mean"],
                s=sizes,
                alpha=0.7,
            )

            ax.set_xlabel(predictor)
            ax.set_ylabel("response_mean")
            ax.set_title(f"{run_name} | {label}")

            out = OUTPUTS / f"{run_name}_{label}_{predictor}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)


def run_relationship_analysis(run_name: str, cfg: dict) -> None:
    df = build_frame(cfg)

    summary = binned_summary(
        df,
        response=cfg["response_name"],
        predictors=cfg["predictors"],
        groups=cfg.get("groups", []),
        bins=cfg.get("bins", 20),
    )

    plot(summary, run_name, cfg.get("groups", []))