"""Build yearly H3 fishing effort feature tables."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from riskscape.config import paths
from riskscape.grid import load_grid


logger = logging.getLogger(__name__)


def input_root() -> Path:
    """Return raw GFW fishing effort root."""
    return paths["raw"] / "gfw"


def output_root() -> Path:
    """Return fishing effort feature output root."""
    return paths["data"] / "features" / "fishing_effort"


def year_from_partition(path: Path) -> int:
    """Return year from a year=YYYY directory."""
    return int(path.name.split("=")[1])


def read_year_file(year_dir: Path) -> pd.DataFrame:
    """Read one yearly fishing effort parquet file."""
    path = year_dir / "fishing_effort.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Fishing effort file not found: {path}")

    return pd.read_parquet(path)


def build_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build point GeoDataFrame from fishing effort records."""
    required = {"date", "hours", "lat", "lon", "vessel_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Fishing effort missing columns: {missing}")

    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    return points


def aggregate_to_h3(df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate fishing effort points to H3 cell and day."""
    if df.empty:
        return pd.DataFrame(
            columns=["h3", "date", "fishing_hours", "vessel_count"]
        )

    points = build_points(df)

    joined = gpd.sjoin(
        points,
        grid[["h3", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return pd.DataFrame(
            columns=["h3", "date", "fishing_hours", "vessel_count"]
        )

    out = (
        joined.groupby(["h3", "date"], as_index=False)
        .agg(
            fishing_hours=("hours", "sum"),
            vessel_count=("vessel_id", "nunique"),
        )
        .sort_values(["date", "h3"])
        .reset_index(drop=True)
    )

    out["h3"] = out["h3"].astype("uint64")
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["fishing_hours"] = out["fishing_hours"].astype("float32")
    out["vessel_count"] = out["vessel_count"].astype("uint16")

    return out


def write_year(df: pd.DataFrame, year: int) -> Path:
    """Write one yearly fishing effort feature table."""
    out_dir = output_root() / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "part.parquet"
    df.to_parquet(out_file, index=False, compression="zstd")

    return out_file


def build_fishing_effort_features() -> list[Path]:
    """Build yearly H3 fishing effort feature tables."""
    grid = load_grid(uint64=True)

    in_root = input_root()
    if not in_root.exists():
        raise FileNotFoundError(f"Fishing effort input root not found: {in_root}")

    outputs = []

    for year_dir in sorted(in_root.glob("year=*")):
        year = year_from_partition(year_dir)

        logger.info("Processing fishing effort year: %s", year)

        df = read_year_file(year_dir)
        features = aggregate_to_h3(df, grid)
        out_file = write_year(features, year)

        logger.info("Saved: %s", out_file)
        logger.info("Rows: %d", len(features))

        outputs.append(out_file)

    return outputs