"""Build yearly H3 species presence feature tables."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid


logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "h3",
    "date",
    "species",
    "presence_count",
    "individual_count",
    "trip_count",
]


def input_file() -> Path:
    """Return telemetry input file."""
    return paths["raw"] / "species_presence" / "saeri_bbal_safs.csv"


def output_root() -> Path:
    """Return species presence feature output root."""
    return paths["data"] / "features" / "species_presence"


def empty_features() -> pd.DataFrame:
    """Return empty species presence feature table."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def read_telemetry() -> pd.DataFrame:
    """Read telemetry CSV."""
    path = input_file()

    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    return pd.read_csv(path, skiprows=1)


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize telemetry records."""
    required = {
        "BirdID_uni",
        "TripNum_uni",
        "species",
        "datetime",
        "lat",
        "lon",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Telemetry missing columns: {missing}")

    out = df.copy()

    out["timestamp"] = pd.to_datetime(
        out["datetime"],
        format="%m/%d/%y %H:%M",
        utc=True,
        errors="coerce",
    )

    invalid_dates = out["timestamp"].isna().sum()
    if invalid_dates:
        logger.warning("Dropping %d rows with invalid datetime", invalid_dates)

    out = out.dropna(subset=["timestamp", "lat", "lon"]).reset_index(drop=True)

    out["date"] = out["timestamp"].dt.floor("D").astype("int64")
    out["lat"] = out["lat"].astype("float32")
    out["lon"] = out["lon"].astype("float32")
    out["species"] = out["species"].astype("string")
    out["BirdID_uni"] = out["BirdID_uni"].astype("string")
    out["TripNum_uni"] = out["TripNum_uni"].astype("string")

    return out


def build_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build point GeoDataFrame from telemetry records."""
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def aggregate_to_h3(
    df: pd.DataFrame,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate telemetry points to H3 cell, day, and species."""
    if df.empty:
        return empty_features()

    points = build_points(df)

    joined = gpd.sjoin(
        points,
        grid[["h3", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return empty_features()

    out = (
        joined.groupby(["h3", "date", "species"], as_index=False)
        .agg(
            presence_count=("BirdID_uni", "count"),
            individual_count=("BirdID_uni", "nunique"),
            trip_count=("TripNum_uni", "nunique"),
        )
        .sort_values(["date", "h3", "species"])
        .reset_index(drop=True)
    )

    out["h3"] = out["h3"].astype("uint64")
    out["date"] = out["date"].astype("int64")
    out["species"] = out["species"].astype("string")
    out["presence_count"] = out["presence_count"].astype("uint16")
    out["individual_count"] = out["individual_count"].astype("uint16")
    out["trip_count"] = out["trip_count"].astype("uint16")

    return out[OUTPUT_COLUMNS]


def write_year(df: pd.DataFrame, year: int) -> Path:
    """Write one yearly species presence feature table."""
    out_dir = output_root() / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "part.parquet"
    df.to_parquet(out_file, index=False, compression="zstd")

    return out_file


def build_species_presence_features() -> list[Path]:
    """Build yearly H3 species presence feature tables."""
    grid = load_grid(uint64=True)

    df = read_telemetry()
    df = clean_telemetry(df)

    features = aggregate_to_h3(df, grid)
    if features.empty:
        logger.info("No species presence records intersected the grid")
        return []

    features["year"] = (
        pd.to_datetime(features["date"], utc=True)
        .dt.year
        .astype("int16")
    )

    outputs = []

    for year, year_features in features.groupby("year", sort=True):
        year_features = year_features.drop(columns="year").copy()
        out_file = write_year(year_features, int(year))

        logger.info("Saved: %s", out_file)
        logger.info("Rows: %d", len(year_features))

        outputs.append(out_file)

    return outputs