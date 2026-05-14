"""Train species-use models with blocked validation splits.

This module keeps the model definitions and metrics from ``train.py`` and only
changes how train/test rows are selected. The split modes follow the main ideas
from blockCV-style validation:

- random: existing row-level random split.
- spatial: hold out spatial H3 parent blocks.
- buffered: hold out spatial H3 parent blocks and remove neighboring cells
  from training.
- environmental_gmm: hold out Bayesian/Gaussian mixture components.
- environmental_seascape: hold out environmental seascape classes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h3
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from riskscape.config import paths
from riskscape.model import train as base_train
from riskscape.model.dataset import FEATURES, SPECIES_TARGET
from riskscape.utils.dates import normalize_date_column


SplitName = Literal[
    "random",
    "spatial",
    "buffered",
    "environmental_gmm",
    "environmental_seascape",
]

BalanceMode = Literal["before", "after", "none"]

RANDOM_STATE = base_train.RANDOM_STATE
# Blocked validation commonly uses smaller holdouts than the production random 75/25 split. 
# A 10% target holdout follows the lower end of the 10-20% range often used for blockCV-style diagnostics while preserving training support.
DEFAULT_TEST_FRACTION = 0.12
DEFAULT_BLOCK_RESOLUTION = 4
DEFAULT_BUFFER_RINGS = 1
DEFAULT_CV_FOLDS = 5
DEFAULT_ENVIRONMENTAL_REGIME_TABLE = "environmental_regimes"
DEFAULT_SEASCAPE_TABLE = "seascapes/som_15x15_hierarchical_k30"
DEFAULT_SEASCAPE_COLUMN = "seascape"
DEFAULT_COMPONENT_TABLE = DEFAULT_ENVIRONMENTAL_REGIME_TABLE

METRICS_DIR = paths["data"] / "modeling" / "metrics"
MODEL_DIR = paths["data"] / "modeling" / "models" / "block_cv"


@dataclass(frozen=True)
class SplitSummary:
    """Record metadata about a validation split."""

    split: str
    heldout_groups: str
    train_rows: int
    test_rows: int
    excluded_buffer_rows: int = 0


def load_partitioned_table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a partitioned modeling table by name."""
    root = paths["data"] / "modeling" / name
    frames = []

    for path in sorted(root.glob("year=*/part.parquet")):
        frames.append(normalize_date_column(pd.read_parquet(path, columns=columns)))

    if not frames:
        raise FileNotFoundError(f"No data found for modeling table: {name}")

    return pd.concat(frames, ignore_index=True)


def scalar_int(value: Any) -> int:
    """Return an int from a pandas scalar value."""
    return int(cast(Any, value))


def load_years_table(
    name: str,
    years: list[int],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load yearly partitions from a modeling table."""
    root = paths["data"] / "modeling" / name
    frames = []

    for year in years:
        path = root / f"year={year}" / "part.parquet"
        if path.exists():
            frames.append(normalize_date_column(pd.read_parquet(path, columns=columns)))

    if not frames:
        raise FileNotFoundError(f"No data found for {name} in years: {years}")

    return pd.concat(frames, ignore_index=True)


def prepare_species_table() -> pd.DataFrame:
    """Load the species training table while preserving H3/date keys."""
    df = load_partitioned_table("species_training")
    required = ["h3", "date", "species", SPECIES_TARGET] + FEATURES
    df = df[required].dropna(subset=["h3", "date", "species", SPECIES_TARGET] + FEATURES)
    df = df.copy()
    df["_y"] = np.log1p(df[SPECIES_TARGET])
    df["_year"] = pd.to_datetime(df["date"], utc=True).dt.year.astype("int16")

    return df


def h3_parent_label(h3_value: int, resolution: int) -> str:
    """Return the parent H3 cell label for a uint64 H3 index."""
    return h3.cell_to_parent(h3.int_to_str(int(h3_value)), resolution)


def add_spatial_blocks(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Add H3 parent-block labels for spatial blocking."""
    out = df.copy()
    unique_h3 = pd.Series(out["h3"].dropna().astype("uint64").unique(), name="h3")
    parent_labels = [
        h3_parent_label(int(value), resolution)
        for value in unique_h3.to_numpy(dtype="uint64")
    ]
    block_lookup = pd.DataFrame(
        {
            "h3": unique_h3,
            "_block_group": parent_labels,
        }
    )

    return out.merge(block_lookup, on="h3", how="left")


def add_gmm_components(df: pd.DataFrame, component_table: str) -> pd.DataFrame:
    """Join Bayesian/Gaussian mixture component labels onto species rows."""
    years = sorted(int(year) for year in df["_year"].unique())
    components = load_years_table(
        component_table,
        years,
        columns=["h3", "date", "bayesian_gmm_k30_component"],
    )
    components = components.drop_duplicates(["h3", "date"])
    components = components.rename(
        columns={"bayesian_gmm_k30_component": "component"}
    )

    out = df.merge(
        components,
        on=["h3", "date"],
        how="left",
        validate="many_to_one",
    )

    if out["component"].isna().any():
        missing = int(out["component"].isna().sum())
        raise ValueError(f"Missing GMM component labels for {missing:,} rows")

    out["_block_group"] = "component_" + out["component"].astype("int16").astype(str)
    return out.drop(columns=["component"])


def add_seascapes(
    df: pd.DataFrame,
    table_name: str,
    seascape_column: str,
) -> pd.DataFrame:
    """Join seascape labels onto species rows."""
    years = sorted(int(year) for year in df["_year"].unique())
    seascapes = load_years_table(
        table_name,
        years,
        columns=["h3", "date", seascape_column],
    )
    seascapes = seascapes.drop_duplicates(["h3", "date"])
    seascapes = seascapes.rename(columns={seascape_column: "seascape"})

    out = df.merge(
        seascapes,
        on=["h3", "date"],
        how="left",
        validate="many_to_one",
    )

    if out["seascape"].isna().any():
        missing = int(out["seascape"].isna().sum())
        raise ValueError(f"Missing seascape labels for {missing:,} rows")

    out["_block_group"] = "seascape_" + out["seascape"].astype("int16").astype(str)
    return out.drop(columns=["seascape"])


def add_split_groups(
    df: pd.DataFrame,
    split: SplitName,
    block_resolution: int,
    seascape_table: str,
    seascape_column: str,
    component_table: str,
) -> pd.DataFrame:
    """Attach split-group labels for blocked split modes."""
    if split in {"random", "buffered", "spatial"}:
        return add_spatial_blocks(df, block_resolution)
    if split == "environmental_gmm":
        return add_gmm_components(df, component_table)
    if split == "environmental_seascape":
        return add_seascapes(df, seascape_table, seascape_column)

    raise ValueError(f"Unknown split: {split}")


def choose_holdout_groups(
    df: pd.DataFrame,
    group_col: str,
    test_fraction: float,
) -> list[str]:
    """Choose whole groups until the requested test fraction is reached."""
    counts = df[group_col].value_counts().sample(frac=1.0, random_state=RANDOM_STATE)
    target = int(round(len(df) * test_fraction))

    heldout: list[str] = []
    total = 0

    for group, count in counts.items():
        heldout.append(str(group))
        total += int(count)
        if total >= target:
            break

    return heldout


def split_random(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split rows randomly into train/test sets."""
    train, test = base_train.split_random(df, test_fraction=test_fraction)
    summary = SplitSummary(
        split="random",
        heldout_groups="random_rows",
        train_rows=len(train),
        test_rows=len(test),
    )

    return train, test, summary


def split_by_group(
    df: pd.DataFrame,
    split: str,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split by complete group labels."""
    heldout = choose_holdout_groups(df, "_block_group", test_fraction)
    heldout_set = set(heldout)
    test_mask = df["_block_group"].isin(heldout_set)

    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()
    summary = SplitSummary(
        split=split,
        heldout_groups=",".join(heldout),
        train_rows=len(train),
        test_rows=len(test),
    )

    return train, test, summary


def buffered_cells(test_h3: pd.Series, buffer_rings: int) -> set[int]:
    """Return H3 cells within the buffer radius around test cells."""
    cells: set[int] = set()

    for value in test_h3.drop_duplicates():
        h3_str = h3.int_to_str(int(value))
        cells.update(h3.str_to_int(cell) for cell in h3.grid_disk(h3_str, buffer_rings))

    return cells


def split_buffered(
    df: pd.DataFrame,
    test_fraction: float,
    buffer_rings: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split by spatial blocks and remove buffered neighbors from training."""
    heldout = choose_holdout_groups(df, "_block_group", test_fraction)
    heldout_set = set(heldout)
    test_mask = df["_block_group"].isin(heldout_set)
    test = df.loc[test_mask].copy()

    buffer = buffered_cells(test["h3"], buffer_rings)
    buffer_mask = df["h3"].isin(buffer)
    train = df.loc[~test_mask & ~buffer_mask].copy()
    excluded = int((~test_mask & buffer_mask).sum())

    summary = SplitSummary(
        split="buffered",
        heldout_groups=",".join(heldout),
        train_rows=len(train),
        test_rows=len(test),
        excluded_buffer_rows=excluded,
    )

    return train, test, summary


def split_table(
    df: pd.DataFrame,
    split: SplitName,
    test_fraction: float,
    buffer_rings: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Dispatch to the requested train/test split."""
    if split == "random":
        return split_random(df, test_fraction)
    if split == "buffered":
        return split_buffered(df, test_fraction, buffer_rings)
    return split_by_group(df, split, test_fraction)


def fold_group_assignments(
    df: pd.DataFrame,
    n_folds: int,
) -> dict[str, int]:
    """Assign complete block groups to row- and species-support-balanced folds."""
    if "_block_group" not in df.columns:
        raise ValueError("Cross-validation folds require _block_group labels")
    if n_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

    species_values = sorted(str(value) for value in df["species"].unique())
    group_stats = (
        df
        .assign(_positive=df[SPECIES_TARGET] > 0)
        .pivot_table(
            index="_block_group",
            columns="species",
            values="_positive",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
    )
    group_stats = group_stats.reindex(columns=species_values, fill_value=0)
    group_stats["rows"] = df["_block_group"].value_counts()
    group_stats["positive_total"] = group_stats[species_values].sum(axis=1)
    group_stats["positive_min"] = group_stats[species_values].min(axis=1)

    if len(group_stats) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from {len(group_stats)} block groups"
        )

    shuffled = group_stats.sample(frac=1.0, random_state=RANDOM_STATE)
    ordered_groups = shuffled.sort_values(
        ["positive_min", "positive_total", "rows"],
        ascending=[False, False, False],
    )

    fold_rows = [0] * n_folds
    fold_counts = [0] * n_folds
    fold_positive = [
        {species: 0 for species in species_values}
        for _ in range(n_folds)
    ]
    target_rows = len(df) / n_folds
    target_positive = {
        species: max(float(df.loc[
            (df["species"] == species) & (df[SPECIES_TARGET] > 0)
        ].shape[0]) / n_folds, 1.0)
        for species in species_values
    }
    max_groups_per_fold = int(np.ceil(len(group_stats) / n_folds))
    assignments: dict[str, int] = {}

    def fold_score(test_fold: int, group: str) -> float:
        rows = fold_rows.copy()
        counts = fold_counts.copy()
        positives = [values.copy() for values in fold_positive]

        rows[test_fold] += scalar_int(group_stats.loc[group, "rows"])
        counts[test_fold] += 1
        for species in species_values:
            positives[test_fold][species] += scalar_int(
                group_stats.loc[group, species]
            )

        row_score = float(np.std(rows) / max(target_rows, 1.0))
        count_score = float(np.std(counts) / max(max_groups_per_fold, 1))
        positive_score = 0.0
        missing_penalty = 0.0

        for species in species_values:
            values = np.array(
                [fold_values[species] for fold_values in positives],
                dtype="float64",
            )
            positive_score += float(
                np.std(values) / target_positive[species]
            )
            missing_penalty += float((values == 0).sum())

        return row_score + positive_score + count_score + (10.0 * missing_penalty)

    for group in ordered_groups.index:
        available = [
            fold
            for fold in range(n_folds)
            if fold_counts[fold] < max_groups_per_fold
        ]
        fold = min(available, key=lambda idx: fold_score(idx, str(group)))
        assignments[str(group)] = fold
        fold_rows[fold] += scalar_int(group_stats.loc[group, "rows"])
        fold_counts[fold] += 1
        for species in species_values:
            fold_positive[fold][species] += scalar_int(
                group_stats.loc[group, species]
            )

    return assignments


def split_cv_fold(
    df: pd.DataFrame,
    split: str,
    fold_assignments: dict[str, int],
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split one cross-validation fold using complete block groups."""
    heldout = sorted(
        group
        for group, group_fold in fold_assignments.items()
        if group_fold == fold
    )
    heldout_set = set(heldout)
    test_mask = df["_block_group"].isin(heldout_set)

    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()
    summary = SplitSummary(
        split=split,
        heldout_groups=",".join(heldout),
        train_rows=len(train),
        test_rows=len(test),
    )

    return train, test, summary


def balance_tables(
    df: pd.DataFrame,
    split: SplitName,
    test_fraction: float,
    buffer_rings: int,
    balance: BalanceMode,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Apply balancing before or after the split."""
    if balance == "before":
        sampled = base_train.sample_training_rows(df)
        return split_table(sampled, split, test_fraction, buffer_rings)

    train, test, summary = split_table(df, split, test_fraction, buffer_rings)

    if balance == "after":
        train = base_train.sample_training_rows(train)
        summary = SplitSummary(
            split=summary.split,
            heldout_groups=summary.heldout_groups,
            train_rows=len(train),
            test_rows=len(test),
            excluded_buffer_rows=summary.excluded_buffer_rows,
        )

    return train, test, summary


def cv_base_table(df: pd.DataFrame, balance: BalanceMode) -> pd.DataFrame:
    """Return the base table used for grouped cross-validation."""
    if balance == "before":
        return base_train.sample_training_rows(df)

    return df.copy()


def split_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    summary: SplitSummary,
) -> pd.DataFrame:
    """Return train/test diagnostics by species and target presence."""
    frames = []

    for split_name, part in [("train", train), ("test", test)]:
        grouped = (
            part
            .assign(_positive=part[SPECIES_TARGET] > 0)
            .groupby("species", observed=True)
            .agg(
                rows=("species", "size"),
                positive_rows=("_positive", "sum"),
                zero_rows=("_positive", lambda values: int((~values).sum())),
                positive_fraction=("_positive", "mean"),
                target_mean=(SPECIES_TARGET, "mean"),
                target_max=(SPECIES_TARGET, "max"),
            )
            .reset_index()
        )
        grouped.insert(0, "partition", split_name)
        frames.append(grouped)

    out = pd.concat(frames, ignore_index=True)
    out["validation_split"] = summary.split
    out["heldout_groups"] = summary.heldout_groups
    out["excluded_buffer_rows"] = summary.excluded_buffer_rows

    return out


def block_group_diagnostics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return row and positive-count diagnostics by block group."""
    if "_block_group" not in df.columns:
        return pd.DataFrame()

    return (
        df
        .assign(_positive=df[SPECIES_TARGET] > 0)
        .groupby(["_block_group", "species"], observed=True)
        .agg(
            rows=("species", "size"),
            positive_rows=("_positive", "sum"),
            zero_rows=("_positive", lambda values: int((~values).sum())),
            positive_fraction=("_positive", "mean"),
            target_mean=(SPECIES_TARGET, "mean"),
            target_max=(SPECIES_TARGET, "max"),
        )
        .reset_index()
        .rename(columns={"_block_group": "block_group"})
        .sort_values(["block_group", "species"])
    )


def save_diagnostics(
    df: pd.DataFrame,
    split: SplitName,
    run_label: str | None,
    suffix: str,
) -> Path:
    """Save split diagnostic table."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    label = f"_{run_label}" if run_label else ""
    path = METRICS_DIR / f"species_model_{split}{label}_{suffix}.csv"
    df.to_csv(path, index=False)
    print("Diagnostics saved:", path)
    return path


def sample_weights(y_train: pd.Series) -> np.ndarray:
    """Return training sample weights."""
    return base_train.sample_weights(y_train)


def dense_float32_matrix(values: Any) -> np.ndarray:
    """Convert dense or sparse sklearn output to a float32 ndarray."""
    if hasattr(values, "toarray"):
        values = values.toarray()

    return np.asarray(values, dtype=np.float32)


def build_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    joint_species: bool,
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame, pd.Series, pd.Series, OneHotEncoder | None]:
    """Build model matrices for joint or single-species models."""
    y_train = train["_y"]
    y_test = test["_y"]

    if not joint_species:
        return train[FEATURES], test[FEATURES], y_train, y_test, None

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    x_train_species = dense_float32_matrix(encoder.fit_transform(train[["species"]]))
    x_test_species = dense_float32_matrix(encoder.transform(test[["species"]]))
    x_train_num = train[FEATURES].to_numpy(dtype="float32")
    x_test_num = test[FEATURES].to_numpy(dtype="float32")
    x_train = np.concatenate((x_train_species, x_train_num), axis=1)
    x_test = np.concatenate((x_test_species, x_test_num), axis=1)

    return x_train, x_test, y_train, y_test, encoder


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    split_summary: SplitSummary,
    model_type: str,
    species_name: str,
    save_models: bool,
) -> dict:
    """Train one model and evaluate it on a blocked test set."""
    joint_species = model_type == "joint_species"
    x_train, x_test, y_train, y_test, encoder = build_xy(train, test, joint_species)

    model = base_train.build_model(model_name)
    model.fit(x_train, y_train, sample_weight=sample_weights(y_train))
    pred_log = model.predict(x_test)
    row = base_train.evaluate_predictions(y_test, pred_log)
    log_row = base_train.metrics(y_test, pred_log)

    row.update(
        {
            "r2_log": log_row["r2"],
            "rmse_log": log_row["rmse"],
            "mae_log": log_row["mae"],
            "model": model_name,
            "model_type": model_type,
            "species": species_name,
            "split": split_summary.split,
            "heldout_groups": split_summary.heldout_groups,
            "train_rows": int(split_summary.train_rows),
            "test_rows": int(split_summary.test_rows),
            "excluded_buffer_rows": int(split_summary.excluded_buffer_rows),
        }
    )

    if save_models:
        suffix = f"{split_summary.split}_{model_type}_{species_name.lower()}"
        model_dir = MODEL_DIR / model_name / suffix
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "encoder": encoder,
            "features": FEATURES,
            "target": SPECIES_TARGET,
            "log_target": True,
            "model_name": model_name,
            "model_type": model_type,
            "species": species_name,
            "split_summary": split_summary,
        }
        joblib.dump(payload, model_dir / "species_model.joblib")

    print(f"{split_summary.split} {model_type} {species_name} {model_name}:")
    print(row)

    return row


def train_joint_models(
    df: pd.DataFrame,
    model_names: list[str],
    split: SplitName,
    test_fraction: float,
    buffer_rings: int,
    balance: BalanceMode,
    save_models: bool,
) -> list[dict]:
    """Train joint-species models under one split strategy."""
    train, test, summary = balance_tables(df, split, test_fraction, buffer_rings, balance)
    rows = []

    for model_name in model_names:
        rows.append(
            train_and_evaluate(
                train=train,
                test=test,
                model_name=model_name,
                split_summary=summary,
                model_type="joint_species",
                species_name="all",
                save_models=save_models,
            )
        )

    return rows


def train_joint_models_cv(
    df: pd.DataFrame,
    model_names: list[str],
    split: SplitName,
    n_folds: int,
    balance: BalanceMode,
    save_models: bool,
) -> list[dict]:
    """Train joint-species models with grouped cross-validation folds."""
    cv_df = cv_base_table(df, balance)
    fold_assignments = fold_group_assignments(cv_df, n_folds)
    rows = []

    for fold in range(n_folds):
        train, test, summary = split_cv_fold(
            cv_df,
            split=split,
            fold_assignments=fold_assignments,
            fold=fold,
        )

        if balance == "after":
            train = base_train.sample_training_rows(train)
            summary = SplitSummary(
                split=summary.split,
                heldout_groups=summary.heldout_groups,
                train_rows=len(train),
                test_rows=len(test),
                excluded_buffer_rows=summary.excluded_buffer_rows,
            )

        for model_name in model_names:
            row = train_and_evaluate(
                train=train,
                test=test,
                model_name=model_name,
                split_summary=summary,
                model_type="joint_species",
                species_name="all",
                save_models=save_models,
            )
            row["cv_fold"] = fold + 1
            row["cv_folds"] = n_folds
            row["actual_test_fraction"] = len(test) / (len(train) + len(test))
            rows.append(row)

    return rows


def train_single_species_models(
    df: pd.DataFrame,
    model_names: list[str],
    split: SplitName,
    test_fraction: float,
    buffer_rings: int,
    balance: BalanceMode,
    save_models: bool,
) -> list[dict]:
    """Train separate species models under one split strategy."""
    rows = []

    for species_name in sorted(df["species"].unique()):
        species_df = df[df["species"] == species_name].copy()
        train, test, summary = balance_tables(
            species_df,
            split,
            test_fraction,
            buffer_rings,
            balance,
        )

        for model_name in model_names:
            rows.append(
                train_and_evaluate(
                    train=train,
                    test=test,
                    model_name=model_name,
                    split_summary=summary,
                    model_type="single_species",
                    species_name=str(species_name),
                    save_models=save_models,
                )
            )

    return rows


def save_metrics(rows: list[dict], split: SplitName, run_label: str | None) -> Path:
    """Save blocked-validation metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_label}" if run_label else ""
    path = METRICS_DIR / f"species_model_{split}{suffix}_block_cv_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Metrics saved:", path)
    return path


def save_cv_summary(rows: list[dict], split: SplitName, run_label: str | None) -> Path:
    """Save mean and standard-deviation summary for grouped CV metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_label}" if run_label else ""
    path = METRICS_DIR / f"species_model_{split}{suffix}_block_cv_summary.csv"
    df = pd.DataFrame(rows)
    metric_cols = [
        "r2",
        "rmse",
        "mae",
        "r2_log",
        "rmse_log",
        "mae_log",
        "actual_test_fraction",
    ]
    summary = (
        df
        .groupby(["model", "model_type", "species"], as_index=False)
        .agg(
            cv_folds=("cv_fold", "count"),
            **{
                f"{col}_{stat}": (col, stat)
                for col in metric_cols
                for stat in ["mean", "std"]
            },
        )
    )
    summary.to_csv(path, index=False)
    print("CV summary saved:", path)
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=[
            "random",
            "spatial",
            "buffered",
            "environmental_gmm",
            "environmental_seascape",
        ],
        default="environmental_seascape",
        help="Validation split strategy.",
    )
    parser.add_argument(
        "--model-type",
        choices=["joint", "single", "both"],
        default="joint",
        help="Train the joint species model, single-species models, or both.",
    )
    parser.add_argument(
        "--models",
        default="extra_trees",
        help="Comma-separated model names. Use 'all' for all configured models.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Held-out fraction. Defaults to 0.12 for blockCV-style diagnostics.",
    )
    parser.add_argument("--block-resolution", type=int, default=DEFAULT_BLOCK_RESOLUTION)
    parser.add_argument("--buffer-rings", type=int, default=DEFAULT_BUFFER_RINGS)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=(
            "Number of grouped cross-validation folds. Use 1 for the existing "
            "single holdout split."
        ),
    )
    parser.add_argument("--component-table", default=DEFAULT_COMPONENT_TABLE)
    parser.add_argument("--seascape-table", default=DEFAULT_SEASCAPE_TABLE)
    parser.add_argument("--seascape-column", default=DEFAULT_SEASCAPE_COLUMN)
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label inserted into the metrics filename.",
    )
    parser.add_argument(
        "--balance",
        choices=["before", "after", "none"],
        default="before",
        help="Balance zeros/positives before split, after split, or not at all.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted blocked-validation model payloads under data/modeling/models/block_cv.",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Create split diagnostics without fitting models.",
    )

    return parser.parse_args()


def model_names_from_arg(value: str) -> list[str]:
    """Resolve model names from a comma-separated CLI argument."""
    if value == "all":
        return list(base_train.MODEL_NAMES)

    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(base_train.MODEL_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    return names


def train_models() -> Path:
    """Train blocked-validation species models."""
    args = parse_args()
    split = cast(SplitName, args.split)
    balance = cast(BalanceMode, args.balance)
    model_names = model_names_from_arg(args.models)

    df = prepare_species_table()
    df = add_split_groups(
        df,
        split=split,
        block_resolution=args.block_resolution,
        seascape_table=args.seascape_table,
        seascape_column=args.seascape_column,
        component_table=args.component_table,
    )

    rows: list[dict] = []

    if args.diagnostics_only:
        if args.cv_folds > 1:
            cv_df = cv_base_table(df, balance)
            fold_assignments = fold_group_assignments(cv_df, args.cv_folds)
            diagnostics = pd.concat(
                [
                    split_diagnostics(
                        *split_cv_fold(
                            cv_df,
                            split=split,
                            fold_assignments=fold_assignments,
                            fold=fold,
                        )
                    ).assign(cv_fold=fold + 1)
                    for fold in range(args.cv_folds)
                ],
                ignore_index=True,
            )
        else:
            train, test, summary = balance_tables(
                df,
                split,
                args.test_fraction,
                args.buffer_rings,
                balance,
            )
            diagnostics = split_diagnostics(train, test, summary)
        save_diagnostics(
            diagnostics,
            split,
            args.run_label,
            "split_diagnostics",
        )
        groups = block_group_diagnostics(df)
        if not groups.empty:
            save_diagnostics(
                groups,
                split,
                args.run_label,
                "block_group_diagnostics",
            )
        return METRICS_DIR / "diagnostics_only"

    if args.cv_folds > 1:
        if args.model_type != "joint":
            raise ValueError("Grouped CV currently supports --model-type joint")
        rows.extend(
            train_joint_models_cv(
                df=df,
                model_names=model_names,
                split=split,
                n_folds=args.cv_folds,
                balance=balance,
                save_models=args.save_models,
            )
        )
        path = save_metrics(rows, split, args.run_label)
        save_cv_summary(rows, split, args.run_label)
        return path

    if args.model_type in {"joint", "both"}:
        rows.extend(
            train_joint_models(
                df=df,
                model_names=model_names,
                split=split,
                test_fraction=args.test_fraction,
                buffer_rings=args.buffer_rings,
                balance=balance,
                save_models=args.save_models,
            )
        )

    if args.model_type in {"single", "both"}:
        rows.extend(
            train_single_species_models(
                df=df,
                model_names=model_names,
                split=split,
                test_fraction=args.test_fraction,
                buffer_rings=args.buffer_rings,
                balance=balance,
                save_models=args.save_models,
            )
        )

    return save_metrics(rows, split, args.run_label)


def main() -> int:
    """Run blocked-validation training."""
    train_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
