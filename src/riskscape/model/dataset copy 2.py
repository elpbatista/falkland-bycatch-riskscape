"""Build model-ready datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from riskscape.grid import load_grid

from riskscape.config import paths


DYNAMIC_FEATURES = [
    "sst",
    "ssh",
    "wind_speed",
    "chl_log",
    "sst_anom",
    "ssh_anom",
    "wind_speed_anom",
    "chl_log_anom",
    "sst_grad",
    "ssh_grad",
    "chl_log_grad",
    "doy_sin",
    "doy_cos",
]

STATIC_FEATURES = [
    "depth_m",
    "slope",
    "dist_coast_m",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
]

FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES

SPECIES_SUPPORT = [
    "presence_count",
    "individual_count",
    "trip_count",
]

FISHING_SUPPORT = [
    "vessel_count",
    "fishing_hours",
]

SPECIES_TARGET = "residence_index"
FISHING_TARGET = "fishing_activity"


def feature_root(table: str) -> Path:
    """Return feature table root."""
    return paths["data"] / "features" / table


def modeling_root(table: str) -> Path:
    """Return modeling table root."""
    return paths["data"] / "modeling" / table


def partition_path(table: str, year: int) -> Path:
    """Return feature partition path."""
    return feature_root(table) / f"year={year}" / "part.parquet"


def output_path(table: str, year: int) -> Path:
    """Return modeling partition path."""
    return modeling_root(table) / f"year={year}" / "part.parquet"


def available_years(table: str) -> list[int]:
    """Return available partition years."""
    root = feature_root(table)
    years = []

    for path in sorted(root.glob("year=*/part.parquet")):
        years.append(int(path.parent.name.split("=")[1]))

    if not years:
        raise FileNotFoundError(f"No partitions found for: {table}")

    return years


def load_partition(table: str, year: int) -> pd.DataFrame:
    """Load one feature partition."""
    path = partition_path(table, year)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_static() -> pd.DataFrame:
    """Load static features and encoded H3 coordinates."""
    path = feature_root("static") / "static.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Static features not found: {path}")

    static = pd.read_parquet(path)

    grid = load_grid(uint64=True)
    validate_columns(grid, ["h3", "lat", "lon"], "grid")

    coords = grid[["h3", "lat", "lon"]].copy()

    lat_rad = np.deg2rad(coords["lat"].astype("float32"))
    lon_rad = np.deg2rad(coords["lon"].astype("float32"))

    coords["lat_sin"] = np.sin(lat_rad).astype("float32")
    coords["lat_cos"] = np.cos(lat_rad).astype("float32")
    coords["lon_sin"] = np.sin(lon_rad).astype("float32")
    coords["lon_cos"] = np.cos(lon_rad).astype("float32")

    out = static.merge(
        coords[
            [
                "h3",
                "lat_sin",
                "lat_cos",
                "lon_sin",
                "lon_cos",
            ]
        ],
        on="h3",
        how="left",
    )

    validate_columns(out, ["h3"] + STATIC_FEATURES, "static")

    return out


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Validate required columns."""
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def save_partition(df: pd.DataFrame, table: str, year: int) -> Path:
    """Save one modeling partition."""
    path = output_path(table, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return path


def join_features(
    df: pd.DataFrame,
    env: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    """Join dynamic and static features."""
    validate_columns(env, ["h3", "date"] + DYNAMIC_FEATURES, "environmental")
    validate_columns(static, ["h3"] + STATIC_FEATURES, "static")

    out = df.merge(
        env[["h3", "date"] + DYNAMIC_FEATURES],
        on=["h3", "date"],
        how="inner",
    )

    out = out.merge(
        static[["h3"] + STATIC_FEATURES],
        on="h3",
        how="left",
    )

    return out


# --- Neighbor index and neighbor-filling functions ---

def neighbor_index_path() -> Path:
    """Return H3 neighbor index path."""
    return paths["data"] / "processed" / "h3_neighbor_index.parquet"


def load_neighbor_index() -> pd.DataFrame:
    """Load H3 neighbor index as h3/neighbor_h3 pairs."""
    path = neighbor_index_path()

    if not path.exists():
        raise FileNotFoundError(f"H3 neighbor index not found: {path}")

    df = pd.read_parquet(path)

    if {"h3", "neighbor_h3"}.issubset(df.columns):
        out = df[["h3", "neighbor_h3"]].copy()
    elif {"h3", "neighbor"}.issubset(df.columns):
        out = df[["h3", "neighbor"]].copy()
        out = out.rename(columns={"neighbor": "neighbor_h3"})
    elif {"h3", "neighbors"}.issubset(df.columns):
        out = df[["h3", "neighbors"]].copy()
        out = out.explode("neighbors")
        out = out.rename(columns={"neighbors": "neighbor_h3"})
    else:
        neighbor_cols = [
            col for col in df.columns
            if col.startswith("neighbor") and col != "neighbors"
        ]

        if not neighbor_cols:
            raise KeyError(
                "Neighbor index must contain h3 plus neighbor_h3, "
                "neighbor, neighbors, or neighbor_* columns"
            )

        out = df[["h3"] + neighbor_cols].melt(
            id_vars="h3",
            value_name="neighbor_h3",
        )
        out = out[["h3", "neighbor_h3"]]

    out = out.dropna(subset=["h3", "neighbor_h3"]).copy()
    out["h3"] = out["h3"].astype("uint64")
    out["neighbor_h3"] = out["neighbor_h3"].astype("uint64")

    return out.drop_duplicates().reset_index(drop=True)


def fill_features_from_neighbors(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Fill missing feature values from same-date H3 neighbors."""
    missing_cols = [col for col in feature_cols if df[col].isna().any()]

    if not missing_cols:
        return df

    out = df.copy()
    neighbors = load_neighbor_index()

    missing_keys = (
        out.loc[out[missing_cols].isna().any(axis=1), ["h3", "date"]]
        .drop_duplicates()
        .copy()
    )

    neighbor_rows = missing_keys.merge(
        neighbors,
        on="h3",
        how="left",
    )

    source = (
        out[["h3", "date"] + missing_cols]
        .drop_duplicates(subset=["h3", "date"])
        .rename(columns={"h3": "neighbor_h3"})
    )

    neighbor_values = neighbor_rows.merge(
        source,
        on=["neighbor_h3", "date"],
        how="left",
    )

    fill_values = (
        neighbor_values
        .groupby(["h3", "date"], as_index=False)[missing_cols]
        .median()
    )

    out = out.merge(
        fill_values,
        on=["h3", "date"],
        how="left",
        suffixes=("", "_fill"),
    )

    for col in missing_cols:
        fill_col = f"{col}_fill"
        out[col] = out[col].fillna(out[fill_col])
        out = out.drop(columns=fill_col)

    remaining = [col for col in feature_cols if out[col].isna().any()]

    if remaining:
        raise ValueError(
            "Remaining NaN values after H3 neighbor filling: "
            f"{remaining}"
        )

    return out


def build_feature_grid(static: pd.DataFrame) -> None:
    """Build filled model feature grid by year."""
    for year in available_years("environmental"):
        env = load_partition("environmental", year)

        if env.empty:
            continue

        base = env[["h3", "date"]].copy()
        df = join_features(base, env, static)
        df = fill_features_from_neighbors(df, FEATURES)

        keep = ["h3", "date"] + FEATURES
        df = df[keep]

        path = save_partition(df, "feature_grid", year)
        print(f"Saved: {path}")
        print(f"Rows: {len(df)}")


def load_feature_grid(year: int) -> pd.DataFrame:
    """Load one filled model feature grid partition."""
    path = output_path("feature_grid", year)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def species_list() -> list[str]:
    """Return species present in species feature table."""
    values = set()

    for year in available_years("species_presence"):
        df = load_partition("species_presence", year)
        validate_columns(df, ["species"], "species_presence")
        values.update(df["species"].dropna().unique().tolist())

    return sorted(values)


def build_species_training(static: pd.DataFrame) -> None:
    """Build species-use training table."""
    for year in available_years("species_presence"):
        species = load_partition("species_presence", year)
        features = load_feature_grid(year)

        if features.empty:
            continue

        if species.empty:
            species = pd.DataFrame(
                columns=["h3", "date", "species"] + SPECIES_SUPPORT
            )
        else:
            validate_columns(
                species,
                ["h3", "date", "species"] + SPECIES_SUPPORT,
                "species_presence",
            )

        species = species.copy()

        # Calculate residence index as presence count divided by individual count^alpha

        alpha = 0.5

        species[SPECIES_TARGET] = (
            species["presence_count"]
            / np.power(species["individual_count"].clip(lower=1), alpha)
        ).astype("float32")

        keep = [
            "h3",
            "date",
            "species",
            SPECIES_TARGET,
        ] + SPECIES_SUPPORT

        observed_species_dates = (
            species[["date", "species"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        grid_dates = features[["h3", "date"]].drop_duplicates()

        base = grid_dates.merge(
            observed_species_dates,
            on="date",
            how="inner",
        )

        df = base.merge(
            species[keep],
            on=["h3", "date", "species"],
            how="left",
        )

        df[SPECIES_TARGET] = df[SPECIES_TARGET].fillna(0.0).astype("float32")
        df["presence_count"] = df["presence_count"].fillna(0).astype("int32")
        df["individual_count"] = df["individual_count"].fillna(0).astype("int32")
        df["trip_count"] = df["trip_count"].fillna(0).astype("int32")

        df = df.merge(
            features[["h3", "date"] + FEATURES],
            on=["h3", "date"],
            how="inner",
        )
        df = df[keep + FEATURES]

        path = save_partition(df, "species_training", year)
        print(f"Saved: {path}")
        print(f"Rows: {len(df)}")



def build_fishing_training(_static: pd.DataFrame) -> None:
    """Build fishing-activity training table."""
    for year in available_years("fishing_effort"):
        fishing = load_partition("fishing_effort", year)
        env = load_partition("environmental", year)

        if env.empty:
            continue

        validate_columns(
            env,
            ["h3", "date"] + DYNAMIC_FEATURES,
            "environmental",
        )

        if fishing.empty:
            fishing = pd.DataFrame(
                columns=["h3", "date"] + FISHING_SUPPORT
            )
        else:
            validate_columns(
                fishing,
                ["h3", "date"] + FISHING_SUPPORT,
                "fishing_effort",
            )

        fishing = fishing.copy()

        # Calculate fishing activity as fishing hours divided by vessel count^alpha

        alpha = 0.5

        fishing[FISHING_TARGET] = (
            fishing["fishing_hours"]
            / np.power(fishing["vessel_count"].clip(lower=1), alpha)
        ).astype("float32")

        keep = ["h3", "date", FISHING_TARGET] + FISHING_SUPPORT

        grid_dates = env[["h3", "date"]].drop_duplicates()
        observed_fishing_dates = (
            fishing[["date"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        base = grid_dates.merge(
            observed_fishing_dates,
            on="date",
            how="inner",
        )

        df = base.merge(
            fishing[keep],
            on=["h3", "date"],
            how="left",
        )

        df[FISHING_TARGET] = df[FISHING_TARGET].fillna(0.0).astype("float32")
        df["vessel_count"] = df["vessel_count"].fillna(0).astype("int32")
        df["fishing_hours"] = df["fishing_hours"].fillna(0).astype("float32")

        df = df[keep]

        path = save_partition(df, "fishing_training", year)
        print(f"Saved: {path}")
        print(f"Rows: {len(df)}")


def build_prediction_grid(static: pd.DataFrame) -> None:
    """Build prediction grid by year."""
    species = pd.DataFrame({"species": species_list()})

    for year in available_years("environmental"):
        df = load_feature_grid(year)

        if df.empty:
            continue

        df = df.merge(species, how="cross")

        keep = ["h3", "date", "species"] + FEATURES
        df = df[keep]

        path = save_partition(df, "prediction_grid", year)
        print(f"Saved: {path}")
        print(f"Rows: {len(df)}")


def build_model_datasets() -> None:
    """Build all model-ready datasets."""
    static = load_static()

    build_feature_grid(static)
    build_species_training(static)
    build_prediction_grid(static)
    build_fishing_training(static)
