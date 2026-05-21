"""Classify environmental seascapes with SOM prototypes and clustering.

This is a SOM-based analogue to dynamic seascape workflows that learn a
high-resolution prototype map first, then aggregate prototypes into a smaller
set of interpretable seascape classes. It does not overwrite the existing
direct k-means seascape products.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import joblib
from minisom import MiniSom
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

from riskscape.config import paths
from riskscape.model.dataset import FEATURES, modeling_root
from riskscape.utils.dates import normalize_date_column


YEARS = "2014-2023"
SOM_X = 15
SOM_Y = 15
N_CLASSES = 15
SAMPLE_PER_YEAR = 200_000
BATCH_ROWS = 250_000
RANDOM_STATE = 42
OUTPUT_ROOT = modeling_root("seascapes")
METRICS_ROOT = paths["data"] / "modeling" / "metrics" / "seascapes"


@dataclass(frozen=True)
class SOMHierarchicalSeascapeModel:
    """Saved SOM-hierarchical seascape model."""

    scaler: StandardScaler
    som: MiniSom
    features: list[str]
    som_x: int
    som_y: int
    n_prototypes: int
    n_classes: int
    prototype_classes: np.ndarray
    prototype_linkage: np.ndarray
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


def component_path(year: int, component_table: str) -> Path:
    """Return Bayesian/GMM component assignment path for one year."""
    return modeling_root(component_table) / f"year={year}" / "part.parquet"


def default_model_name(som_x: int, som_y: int, n_classes: int) -> str:
    """Return default output model name."""
    return f"som_{som_x}x{som_y}_hierarchical_k{n_classes}"


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


def parse_int_list(values: str) -> list[int]:
    """Parse comma-separated integers."""
    parsed = sorted({int(item.strip()) for item in values.split(",") if item.strip()})
    if not parsed:
        raise ValueError("No integer values selected")
    return parsed


def year_label(years: list[int]) -> str:
    """Return display-safe selected-year text."""
    if len(years) == 1:
        return str(years[0])
    if years == list(range(min(years), max(years) + 1)):
        return f"{min(years)}-{max(years)}"
    return "_".join(str(year) for year in years)


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
    """Load and sample feature rows for SOM fitting."""
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


def prototype_index(xy: tuple[int, int], som_y: int) -> int:
    """Return flat prototype index from a SOM winner coordinate."""
    return (int(xy[0]) * som_y) + int(xy[1])


def fit_model(
    years: list[int],
    som_x: int,
    som_y: int,
    n_classes: int,
    sample_per_year: int,
    random_state: int,
    out_model_name: str,
    iterations: int,
    sigma: float,
    learning_rate: float,
) -> Path:
    """Fit a SOM and cluster its prototypes into seascape classes."""
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

    som = MiniSom(
        x=som_x,
        y=som_y,
        input_len=len(features),
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function="gaussian",
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=random_state,
    )
    som.pca_weights_init(x_scaled)
    som.train_random(x_scaled, iterations, verbose=True)

    weights = som.get_weights().reshape(som_x * som_y, len(features))
    tree = linkage(pdist(weights, metric="euclidean"), method="ward")
    classes = fcluster(tree, t=n_classes, criterion="maxclust").astype("int16") - 1

    payload = SOMHierarchicalSeascapeModel(
        scaler=scaler,
        som=som,
        features=features,
        som_x=som_x,
        som_y=som_y,
        n_prototypes=som_x * som_y,
        n_classes=n_classes,
        prototype_classes=classes,
        prototype_linkage=tree,
        random_state=random_state,
    )

    out_file = model_path(out_model_name)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_file)
    print(f"Saved model: {out_file}")
    print(f"Quantization error: {som.quantization_error(x_scaled):.6f}")
    print(f"Topographic error: {som.topographic_error(x_scaled):.6f}")
    return out_file


def iter_batches(df: pd.DataFrame, batch_rows: int) -> Iterable[pd.DataFrame]:
    """Yield dataframe batches."""
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start:start + batch_rows].copy()


def assign_batch(batch: pd.DataFrame, payload: SOMHierarchicalSeascapeModel) -> pd.DataFrame:
    """Assign one feature batch to SOM prototypes and final classes."""
    features = payload.features
    x = batch[features].to_numpy(dtype="float64")
    x_scaled = payload.scaler.transform(x)
    weights = payload.som.get_weights().reshape(
        payload.n_prototypes,
        len(features),
    )
    prototype_ids, distances = pairwise_distances_argmin_min(x_scaled, weights)

    out = batch[["h3", "date"]].copy()
    out["som_prototype"] = prototype_ids.astype("int16")
    out["seascape"] = payload.prototype_classes[prototype_ids].astype("int16")
    out["seascape_distance"] = distances.astype("float32")
    return out


def assign_year(
    year: int,
    payload: SOMHierarchicalSeascapeModel,
    out_model_name: str,
    batch_rows: int,
) -> Path:
    """Assign SOM-hierarchical seascapes for one feature-grid year."""
    path = feature_grid_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Feature-grid partition not found: {path}")

    columns = ["h3", "date", *payload.features]
    df = normalize_date_column(pd.read_parquet(path, columns=columns))
    df = clean_features(df, payload.features)

    frames = [
        assign_batch(batch, payload)
        for batch in iter_batches(df, batch_rows)
    ]
    out = normalize_date_column(pd.concat(frames, ignore_index=True))

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


def summarize_seascapes(years: list[int], out_model_name: str) -> Path:
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
            count(DISTINCT s.som_prototype) AS n_som_prototypes,
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


def compare_hierarchical_cuts(
    out_model_name: str,
    candidate_classes: list[int],
) -> Path:
    """Compare alternative hierarchical cuts of the fitted SOM prototypes."""
    payload = joblib.load(model_path(out_model_name))
    weights = payload.som.get_weights().reshape(payload.n_prototypes, -1)
    rows: list[dict[str, float | int | str]] = []

    for n_classes in candidate_classes:
        if n_classes < 2:
            raise ValueError("Candidate class counts must be at least 2")
        if n_classes >= payload.n_prototypes:
            raise ValueError(
                f"Candidate class count {n_classes} must be below "
                f"{payload.n_prototypes} prototypes"
            )

        labels = fcluster(
            payload.prototype_linkage,
            t=n_classes,
            criterion="maxclust",
        ) - 1
        counts = np.bincount(labels, minlength=n_classes)
        centroids = np.vstack(
            [weights[labels == label].mean(axis=0) for label in range(n_classes)]
        )
        within_ss = float(
            sum(
                np.square(weights[labels == label] - centroids[label]).sum()
                for label in range(n_classes)
            )
        )

        rows.append(
            {
                "model": out_model_name,
                "som_x": payload.som_x,
                "som_y": payload.som_y,
                "n_prototypes": payload.n_prototypes,
                "n_classes": n_classes,
                "min_prototypes_per_class": int(counts.min()),
                "median_prototypes_per_class": float(np.median(counts)),
                "max_prototypes_per_class": int(counts.max()),
                "prototype_balance_ratio": float(counts.min() / counts.max()),
                "singleton_classes": int((counts == 1).sum()),
                "within_prototype_ss": within_ss,
                "mean_within_prototype_ss": within_ss / payload.n_prototypes,
                "silhouette_score": float(silhouette_score(weights, labels)),
                "calinski_harabasz_score": float(
                    calinski_harabasz_score(weights, labels)
                ),
                "davies_bouldin_score": float(davies_bouldin_score(weights, labels)),
                "mean_centroid_distance": float(pdist(centroids).mean()),
            }
        )

    out = pd.DataFrame(rows)
    out_file = METRICS_ROOT / f"seascape_som_cut_comparison_{out_model_name}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"Saved SOM cut comparison: {out_file}")
    return out_file


def export_hierarchical_cut_assignments(
    years: list[int],
    source_model_name: str,
    out_model_name: str,
    n_classes: int,
) -> list[Path]:
    """Export assignments for another cut using saved SOM prototype IDs."""
    payload = joblib.load(model_path(source_model_name))
    if n_classes < 2 or n_classes >= payload.n_prototypes:
        raise ValueError(
            f"n_classes must be between 2 and {payload.n_prototypes - 1}"
        )

    labels = (
        fcluster(
            payload.prototype_linkage,
            t=n_classes,
            criterion="maxclust",
        ) - 1
    ).astype("int16")
    mapping = pd.DataFrame(
        {
            "som_prototype": np.arange(payload.n_prototypes, dtype="int16"),
            "seascape": labels,
        }
    )

    out_files: list[Path] = []
    for year in years:
        source_file = seascape_path(year, source_model_name)
        if not source_file.exists():
            raise FileNotFoundError(
                f"Source SOM assignment partition not found: {source_file}"
            )

        out_file = seascape_path(year, out_model_name)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        query = """
            COPY (
                SELECT
                    s.h3,
                    s.date,
                    s.som_prototype,
                    CAST(m.seascape AS SMALLINT) AS seascape,
                    s.seascape_distance
                FROM read_parquet($source_file) AS s
                INNER JOIN mapping AS m
                    USING (som_prototype)
            )
            TO $out_file (FORMAT PARQUET)
        """

        with duckdb.connect(database=":memory:") as con:
            con.register("mapping", mapping)
            con.execute(
                query,
                {
                    "source_file": str(source_file),
                    "out_file": str(out_file),
                },
            )

        out_files.append(out_file)
        print(f"Saved cut assignment: {out_file}")

    return out_files


def prototype_summary(out_model_name: str) -> Path:
    """Write SOM prototype-to-seascape lookup."""
    payload = joblib.load(model_path(out_model_name))
    rows = []
    weights = payload.som.get_weights().reshape(
        payload.n_prototypes,
        len(payload.features),
    )
    for idx, seascape in enumerate(payload.prototype_classes):
        x = idx // payload.som_y
        y = idx % payload.som_y
        row = {
            "som_prototype": idx,
            "som_x": x,
            "som_y": y,
            "seascape": int(seascape),
        }
        row.update(
            {
                f"{feature}_scaled": float(value)
                for feature, value in zip(payload.features, weights[idx])
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out_file = METRICS_ROOT / f"seascape_som_prototypes_{out_model_name}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_file, index=False)
    print(f"Saved prototype summary: {out_file}")
    return out_file


def compare_with_cube_components(
    years: list[int],
    out_model_name: str,
    component_table: str,
    component_column: str,
) -> tuple[Path, Path] | None:
    """Compare seascape classes with Bayesian/GMM component assignments."""
    if not component_column.replace("_", "").isalnum():
        raise ValueError(f"Unsafe component column name: {component_column}")

    available = [
        year
        for year in years
        if seascape_path(year, out_model_name).exists()
        and component_path(year, component_table).exists()
    ]
    missing = sorted(set(years) - set(available))
    if missing:
        print(
            "Skipping component comparison for missing year partitions: "
            + ", ".join(str(year) for year in missing)
        )

    if not available:
        print(
            "No overlapping seascape/component partitions found; "
            "skipping component comparison."
        )
        return None

    seascape_files = [str(seascape_path(year, out_model_name)) for year in available]
    component_files = [str(component_path(year, component_table)) for year in available]

    query = f"""
        WITH cube AS (
            SELECT
                DISTINCT h3,
                CAST(date AS DATE) AS date,
                {component_column} AS component
            FROM read_parquet($component_files)
        )
        SELECT
            s.seascape,
            c.component,
            count(*) AS n_cell_days
        FROM (
            SELECT h3, CAST(date AS DATE) AS date, seascape
            FROM read_parquet($seascape_files)
        ) AS s
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
                "compared_years": year_label(available),
                "model": out_model_name,
                "component_table": component_table,
                "component_column": component_column,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default=YEARS)
    parser.add_argument("--som-x", type=int, default=SOM_X)
    parser.add_argument("--som-y", type=int, default=SOM_Y)
    parser.add_argument("--n-classes", type=int, default=N_CLASSES)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--sample-per-year", type=int, default=SAMPLE_PER_YEAR)
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--assign", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--prototype-summary", action="store_true")
    parser.add_argument("--compare-components", action="store_true")
    parser.add_argument("--compare-cuts", action="store_true")
    parser.add_argument("--export-cut", action="store_true")
    parser.add_argument(
        "--source-model-name",
        default=None,
        help="Existing SOM model/assignment name to recut for --export-cut.",
    )
    parser.add_argument(
        "--candidate-classes",
        default="8,10,12,15,18,20,30",
        help="Comma-separated class counts for --compare-cuts.",
    )
    parser.add_argument(
        "--component-table",
        default="environmental_regimes",
    )
    parser.add_argument("--component-column", default="bayesian_gmm_k30_component")
    parser.add_argument("--all-steps", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run SOM-hierarchical seascape classification workflow."""
    args = parse_args()
    years = parse_years(args.years)
    out_model_name = args.model_name or default_model_name(
        args.som_x,
        args.som_y,
        args.n_classes,
    )

    run_fit = args.fit or args.all_steps
    run_assign = args.assign or args.all_steps
    run_summarize = args.summarize or args.all_steps
    run_prototype_summary = args.prototype_summary or args.all_steps
    run_compare = args.compare_components or args.all_steps
    run_compare_cuts = args.compare_cuts
    run_export_cut = args.export_cut

    if not any(
        [
            run_fit,
            run_assign,
            run_summarize,
            run_prototype_summary,
            run_compare,
            run_compare_cuts,
            run_export_cut,
        ]
    ):
        raise ValueError(
            "Select at least one action: --fit, --assign, --summarize, "
            "--prototype-summary, --compare-components, --compare-cuts, --export-cut, "
            "or --all-steps."
        )

    if run_fit:
        fit_model(
            years=years,
            som_x=args.som_x,
            som_y=args.som_y,
            n_classes=args.n_classes,
            sample_per_year=args.sample_per_year,
            random_state=RANDOM_STATE,
            out_model_name=out_model_name,
            iterations=args.iterations,
            sigma=args.sigma,
            learning_rate=args.learning_rate,
        )

    if run_assign:
        assign_years(
            years=years,
            out_model_name=out_model_name,
            batch_rows=args.batch_rows,
        )

    if run_summarize:
        summarize_seascapes(years=years, out_model_name=out_model_name)

    if run_prototype_summary:
        prototype_summary(out_model_name=out_model_name)

    if run_compare:
        compare_with_cube_components(
            years=years,
            out_model_name=out_model_name,
            component_table=args.component_table,
            component_column=args.component_column,
        )

    if run_compare_cuts:
        compare_hierarchical_cuts(
            out_model_name=out_model_name,
            candidate_classes=parse_int_list(args.candidate_classes),
        )

    if run_export_cut:
        export_hierarchical_cut_assignments(
            years=years,
            source_model_name=args.source_model_name or out_model_name,
            out_model_name=out_model_name,
            n_classes=args.n_classes,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
