#!/usr/bin/env python3
"""Export weekly H3 risk layers for Mapbox.

The output is geographic GeoJSONL because Mapbox uploads require geometry.
Properties are intentionally short to reduce repeated text in large files:

- ``h``: uint64 H3 id
- ``su``: ungated species-use log prediction
- ``pl``: plausibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = (
    PROJECT_ROOT
    / "data"
    / "grids"
    / "h3_res6_falkland_islands_uint64.parquet"
)
DEFAULT_WEEKLY_DIR = (
    PROJECT_ROOT
    / "data"
    / "plot_exports"
    / "dashboard"
    / "mapbox"
    / "risk_weekly"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "plot_exports"
    / "dashboard"
    / "mapbox"
    / "risk_weekly_geojson"
)


def round_coordinates(value: Any, precision: int) -> Any:
    """Round nested coordinate arrays to the requested decimal precision."""
    if isinstance(value, float):
        return round(value, precision)
    if isinstance(value, list):
        return [round_coordinates(item, precision) for item in value]
    if isinstance(value, tuple):
        return [round_coordinates(item, precision) for item in value]
    return value


def rounded_geometry(geometry: Any, precision: int) -> dict[str, Any]:
    """Return a GeoJSON geometry with rounded coordinates."""
    geojson = geometry.__geo_interface__
    out = {
        "type": geojson["type"],
        "coordinates": round_coordinates(geojson["coordinates"], precision),
    }
    return out


def load_weekly_table(path: Path, weeks: set[int]) -> pd.DataFrame:
    """Load and filter one species weekly table."""
    if not path.exists():
        raise FileNotFoundError(path)

    table = pd.read_parquet(path)
    table = table[table["iso_week"].isin(weeks)].copy()
    table["h3"] = table["h3"].astype("uint64")
    table["iso_week"] = table["iso_week"].astype("uint8")
    return table


def clean_properties(row: Any, value_precision: int) -> dict[str, Any]:
    """Return compact Mapbox feature properties."""
    return {
        "h": int(row.h3),
        "su": round(float(row.species_use_log_pred), value_precision),
        "pl": round(float(row.plausibility), value_precision),
    }


def write_geojsonl(
    gdf: gpd.GeoDataFrame,
    path: Path,
    coordinate_precision: int,
    value_precision: int,
) -> None:
    """Write compact line-delimited GeoJSON features."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in gdf.itertuples(index=False):
            feature = {
                "type": "Feature",
                "id": int(row.h3),
                "properties": clean_properties(row, value_precision),
                "geometry": rounded_geometry(row.geometry, coordinate_precision),
            }
            file.write(json.dumps(feature, separators=(",", ":")) + "\n")


def export_weekly_risk(
    grid_path: Path,
    weekly_path: Path,
    output_path: Path,
    weeks: set[int],
    coordinate_precision: int,
    value_precision: int,
) -> None:
    """Join weekly values to H3 geometry and write GeoJSONL."""
    grid = gpd.read_parquet(grid_path)
    if grid.crs is None:
        raise ValueError(f"Missing CRS: {grid_path}")
    grid = grid.to_crs("EPSG:4326")[["h3", "geometry"]].copy()
    grid["h3"] = grid["h3"].astype("uint64")

    weekly = load_weekly_table(weekly_path, weeks=weeks)
    out = grid.merge(weekly, on="h3", how="inner").sort_values(["iso_week", "h3"])
    if out.empty:
        raise ValueError(f"No rows found for requested weeks: {sorted(weeks)}")

    write_geojsonl(
        out,
        output_path,
        coordinate_precision=coordinate_precision,
        value_precision=value_precision,
    )
    print(f"{output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"rows={len(out):,}")


def parse_weeks(value: str) -> set[int]:
    """Parse comma-separated weeks or ranges like 1,2,10-13."""
    weeks: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            weeks.update(range(int(start), int(end) + 1))
        else:
            weeks.add(int(part))
    invalid = [week for week in weeks if week < 1 or week > 53]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid ISO weeks: {invalid}")
    return weeks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--weekly-dir", type=Path, default=DEFAULT_WEEKLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--species", required=True, choices=["BBAL", "SAFS"])
    parser.add_argument("--weeks", type=parse_weeks, required=True)
    parser.add_argument("--coordinate-precision", type=int, default=6)
    parser.add_argument("--value-precision", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Run the export."""
    args = parse_args()
    weekly_path = args.weekly_dir / f"{args.species}_weekly.parquet"
    week_label = "_".join(f"w{week:02d}" for week in sorted(args.weeks))
    output_path = args.output_dir / f"{args.species}_{week_label}.geojsonl"
    export_weekly_risk(
        grid_path=args.grid,
        weekly_path=weekly_path,
        output_path=output_path,
        weeks=args.weeks,
        coordinate_precision=args.coordinate_precision,
        value_precision=args.value_precision,
    )


if __name__ == "__main__":
    main()
