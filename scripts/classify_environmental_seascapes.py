"""Classify environmental seascapes and compare them with GMM components."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, modeling_root


N_CLASSES = 10
YEARS = "2014-2023"
SAMPLE_PER_YEAR = 200_000
BATCH_ROWS = 250_000
RANDOM_STATE = 42
MODEL_NAME = "kmeans_k10"
OUTPUT_ROOT = modeling_root("seascapes")
METRICS_ROOT = paths["data"] / "modeling" / "metrics" / "seascapes"


@dataclass(frozen=True)
class SeascapeModel:
    """Saved seascape classification model."""

    scaler: StandardScaler
    model: MiniBatchKMeans
    features: list[str]
    n_classes: int
    random_state: int


def feature_grid_path(year: int) -> Path:
    """Return one feature-grid partition path."""
    return modeling_root("feature_grid") / f"year={year}" / "part.parquet"


def seascape_root(model_name: str) -> Path:
    """Return seascape assignment root for a model name."""
    return OUTPUT_ROOT / model_name


def seascape_path(year: int, model_name: str) -> Path:
    """Return seascape assignment partition path for one year."""
    return seascape_root(model_name) / f"year={year}" / "part.parquet"


def model_path(model_name: str) -> Path:
    """Return fitted seascape model path."""
    return paths["data"] / "modeling" / "models" / "seascapes" / f"{model_name}.joblib"


def component_path(year: int) -> Path:
    """Return Bayesian/GMM component assignment path for one year."""
    return modeling_root("cube_components") / f"year={year}" / "part.parquet"


def available_years() -> list[int]:
    """Return years with feature-grid partitions."""
    years: list[int] = []

    for path in sorted(modeling_root("feature_grid").glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=", maxsplit=1)[1]))

    if not years:
        raise FileNotFoundError("No feature_grid partitions found")

    return years


def parse_years(years: str) -> list[int]:
    """Parse all, one year, a range, or comma-separated years."""
    if years.lower() == "all":
        return available_years()

    parsed: set[int] = set()
    for part in years.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            parsed.update(range(int(start_text), int(end_text) + 1))
        else:
            parsed.add(int(item))

    if not parsed:
        raise ValueError("No years selected")

    return sorted(parsed)


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if len(years) == 1:
        return str(years[0])
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"
    return "_".join(str(year) for year in years)


def model_name(n_classes: int) -> str:
    """Return default model name for a class count."""
    return f"kmeans_k{n_classes}"


def clean_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Drop rows with missing or infinite feature values."""
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    return out.dropna(subset=features)


def sample_feature_grid(
    year: int,
    features: list[str],
    sample_per_year: int,
    random_state: int,
) -> pd.DataFrame:
    """Load and sample feature rows for model fitting."""
    path = feature_grid_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Feature-grid partition not found: {path}")

    df = pd.read_parquet(path, columns=features)
    df = clean_features(df, features)

    if len(df) > sample_per_year:
        df = df.sample(
            n=sample_per_year,
            random_state=random_state + year,
        )

    return df.reset_index(drop=True)


def fit_seascape_model(
    years: list[int],
    n_classes: int,
    sample_per_year: int,
    random_state: int,
    out_model_name: str,
) -> Path:
    """Fit a feature-only seascape classifier."""
    features = list(FEATURES)
    samples = [
        sample_feature_grid(
            year,
            features=features,
            sample_per_year=sample_per_year,
            random_state=random_state,
        )
        for year in years
    ]
    training = pd.concat(samples, ignore_index=True)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(training[features].to_numpy(dtype="float64"))

    model = MiniBatchKMeans(
        n_clusters=n_classes,
        random_state=random_state,
        batch_size=8192,
        n_init="auto",
    )
    model.fit(x_scaled)

    payload = SeascapeModel(
        scaler=scaler,
        model=model,
        features=features,
        n_classes=n_classes,
        random_state=random_state,
    )

    out_file = model_path(out_model_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_file)

    print(f"Saved model: {out_file}")

    return out_file


def iter_batches(df: pd.DataFrame, batch_rows: int) -> Iterable[pd.DataFrame]:
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def assign_batch(batch: pd.DataFrame, payload: SeascapeModel) -> pd.DataFrame:
    """Assign seascape class labels to a batch."""
    features = payload.features
    x = batch[features].to_numpy(dtype="float64")
    x_scaled = payload.scaler.transform(x)
    labels, distances = pairwise_distances_argmin_min(
        x_scaled,
        payload.model.cluster_centers_,
    )

    out = batch[["h3", "date"]].copy()
    out["seascape"] = labels.astype("int16")
    out["seascape_distance"] = distances.astype("float32")

    return out


def assign_year(
    year: int,
    payload: SeascapeModel,
    out_model_name: str,
    batch_rows: int,
) -> Path:
    """Assign seascapes for one feature-grid year."""
    path = feature_grid_path(year)

    if not path.exists():
        raise FileNotFoundError(f"Feature-grid partition not found: {path}")

    columns = ["h3", "date", *payload.features]
    df = pd.read_parquet(path, columns=columns)
    df = clean_features(df, payload.features)

    frames = [
        assign_batch(batch, payload)
        for batch in iter_batches(df, batch_rows)
    ]
    out = pd.concat(frames, ignore_index=True)

    out_file = seascape_path(year, out_model_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False, compression="zstd")

    print(f"Saved assignments: {out_file}")

    return out_file


def assign_years(
    years: list[int],
    out_model_name: str,
    batch_rows: int,
) -> list[Path]:
    """Assign seascapes for selected years."""
    payload = joblib.load(model_path(out_model_name))
    return [
        assign_year(
            year=year,
            payload=payload,
            out_model_name=out_model_name,
            batch_rows=batch_rows,
        )
        for year in years
    ]


def summarize_seascapes(
    years: list[int],
    out_model_name: str,
) -> Path:
    """Summarize environmental feature values by seascape class."""
    feature_files = [str(feature_grid_path(year)) for year in years]
    seascape_files = [str(seascape_path(year, out_model_name)) for year in years]
    selected = ", ".join(
        f"avg(f.{feature}) AS {feature}_mean, "
        f"stddev_samp(f.{feature}) AS {feature}_sd"
        for feature in FEATURES
    )

    query = f"""
        SELECT
            s.seascape,
            count(*) AS n_cell_days,
            {selected}
        FROM read_parquet($seascape_files) AS s
        INNER JOIN read_parquet($feature_files) AS f
            USING (h3, date)
        GROUP BY s.seascape
        ORDER BY s.seascape
    """

    with duckdb.connect(database=":memory:") as con:
        summary = con.execute(
            query,
            {
                "seascape_files": seascape_files,
                "feature_files": feature_files,
            },
        ).fetchdf()

    out_file = (
        METRICS_ROOT
        / f"seascape_summary_{out_model_name}_{year_label(years)}.csv"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_file, index=False)

    print(f"Saved summary: {out_file}")

    return out_file


def compare_with_cube_components(
    years: list[int],
    out_model_name: str,
) -> tuple[Path, Path]:
    """Compare seascape classes with Bayesian/GMM component assignments."""
    seascape_files = [str(seascape_path(year, out_model_name)) for year in years]
    component_files = [str(component_path(year)) for year in years]

    query = """
        WITH cube AS (
            SELECT DISTINCT h3, date, component
            FROM read_parquet($component_files)
        )
        SELECT
            s.seascape,
            c.component,
            count(*) AS n_cell_days
        FROM read_parquet($seascape_files) AS s
        INNER JOIN cube AS c
            USING (h3, date)
        GROUP BY s.seascape, c.component
        ORDER BY s.seascape, c.component
    """

    with duckdb.connect(database=":memory:") as con:
        crosswalk = con.execute(
            query,
            {
                "seascape_files": seascape_files,
                "component_files": component_files,
            },
        ).fetchdf()

    total = int(crosswalk["n_cell_days"].sum())
    seascape_max = int(crosswalk.groupby("seascape")["n_cell_days"].max().sum())
    component_max = int(crosswalk.groupby("component")["n_cell_days"].max().sum())

    metrics = pd.DataFrame(
        [
            {
                "years": year_label(years),
                "model": out_model_name,
                "n_cell_days": total,
                "seascape_to_component_purity": seascape_max / total,
                "component_to_seascape_purity": component_max / total,
            }
        ]
    )

    crosswalk_file = (
        METRICS_ROOT
        / f"seascape_component_crosswalk_{out_model_name}_{year_label(years)}.csv"
    )
    metrics_file = (
        METRICS_ROOT
        / f"seascape_component_comparison_{out_model_name}_{year_label(years)}.csv"
    )
    crosswalk_file.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(crosswalk_file, index=False)
    metrics.to_csv(metrics_file, index=False)

    print(f"Saved crosswalk: {crosswalk_file}")
    print(f"Saved comparison metrics: {metrics_file}")

    return crosswalk_file, metrics_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit feature-only environmental seascape classes, assign them to "
            "the H3/date cube, and compare them with Bayesian/GMM components."
        )
    )
    parser.add_argument(
        "--years",
        default=YEARS,
        help="Years to process: all, one year, a range, or comma-separated years.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=N_CLASSES,
        help="Number of seascape classes.",
    )
    parser.add_argument(
        "--sample-per-year",
        type=int,
        default=SAMPLE_PER_YEAR,
        help="Maximum feature-grid rows sampled per year for fitting.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=BATCH_ROWS,
        help="Rows per assignment batch.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional output model name. Defaults to kmeans_k<n_classes>.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the seascape model.",
    )
    parser.add_argument(
        "--assign",
        action="store_true",
        help="Assign fitted seascape classes to selected years.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize feature values by seascape class.",
    )
    parser.add_argument(
        "--compare-components",
        action="store_true",
        help="Compare seascape classes against Bayesian/GMM cube components.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Run fit, assign, summarize, and compare.",
    )

    return parser.parse_args()


def main() -> int:
    """Run seascape classification workflow."""
    args = parse_args()
    years = parse_years(args.years)
    out_model_name = args.model_name or model_name(args.n_classes)

    run_fit = args.fit or args.all_steps
    run_assign = args.assign or args.all_steps
    run_summarize = args.summarize or args.all_steps
    run_compare = args.compare_components or args.all_steps

    if not any([run_fit, run_assign, run_summarize, run_compare]):
        raise ValueError(
            "Select at least one action: --fit, --assign, --summarize, "
            "--compare-components, or --all-steps."
        )

    if run_fit:
        fit_seascape_model(
            years=years,
            n_classes=args.n_classes,
            sample_per_year=args.sample_per_year,
            random_state=RANDOM_STATE,
            out_model_name=out_model_name,
        )

    if run_assign:
        assign_years(
            years=years,
            out_model_name=out_model_name,
            batch_rows=args.batch_rows,
        )

    if run_summarize:
        summarize_seascapes(
            years=years,
            out_model_name=out_model_name,
        )

    if run_compare:
        compare_with_cube_components(
            years=years,
            out_model_name=out_model_name,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
