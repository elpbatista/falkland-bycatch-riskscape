#!/usr/bin/env python3
"""Export the H3 grid as Mapbox-ready geometry inputs.

The dashboard keeps daily risk values in separate attribute tables. This
script exports only the stable H3 geometry and the existing uint64 H3 key used
throughout the project.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "plot_exports" / "dashboard" / "mapbox"


def clean_feature(row: Any) -> dict[str, Any]:
    """Return a minimal H3 geometry feature for Mapbox."""
    h3_value = int(row.h3)

    return {
        "type": "Feature",
        "id": h3_value,
        "properties": {
            "h3": h3_value,
            "lat": row.lat,
            "lon": row.lon,
        },
        "geometry": row.geometry.__geo_interface__,
    }


def write_geojson(features: list[dict[str, Any]], path: Path) -> None:
    """Write a FeatureCollection."""
    payload = {
        "type": "FeatureCollection",
        "features": features,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def write_geojsonl(features: list[dict[str, Any]], path: Path) -> None:
    """Write line-delimited GeoJSON features."""
    with path.open("w") as file:
        for feature in features:
            file.write(json.dumps(feature, separators=(",", ":")) + "\n")


def write_recipe(path: Path, source_uri: str, layer_name: str, minzoom: int, maxzoom: int) -> None:
    """Write a minimal Mapbox Tiling Service vector recipe."""
    recipe = {
        "version": 1,
        "layers": {
            layer_name: {
                "source": source_uri,
                "minzoom": minzoom,
                "maxzoom": maxzoom,
                "features": {
                    "id": ["get", "h3"],
                },
            },
        },
    }
    path.write_text(json.dumps(recipe, indent=2) + "\n")


def export_grid(
    grid_path: Path,
    output_dir: Path,
    source_uri: str,
    layer_name: str,
    minzoom: int,
    maxzoom: int,
) -> None:
    """Export cleaned geometry, lookup table, and recipe."""
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_parquet(grid_path)
    if grid.crs is None:
        raise ValueError(f"Missing CRS: {grid_path}")
    grid = grid.to_crs("EPSG:4326").sort_values("h3").reset_index(drop=True)
    features = [clean_feature(row) for row in grid.itertuples(index=False)]

    geojson_path = output_dir / "h3_res6_geometry.geojson"
    geojsonl_path = output_dir / "h3_res6_geometry.geojsonl"
    lookup_path = output_dir / "h3_res6_geometry_lookup.csv"
    recipe_path = output_dir / "h3_res6_geometry_mts_recipe.json"

    write_geojson(features, geojson_path)
    write_geojsonl(features, geojsonl_path)
    pd.DataFrame(
        [feature["properties"] for feature in features],
        columns=["h3", "lat", "lon"],
    ).to_csv(lookup_path, index=False)
    write_recipe(recipe_path, source_uri, layer_name, minzoom, maxzoom)

    for path in [geojson_path, geojsonl_path, lookup_path, recipe_path]:
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"{path} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-uri",
        default="mapbox://tileset-source/{username}/falkland-h3-res6-geometry",
        help="MTS source URI placeholder or final source URI.",
    )
    parser.add_argument("--layer-name", default="h3_res6")
    parser.add_argument("--minzoom", type=int, default=4)
    parser.add_argument("--maxzoom", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run the export."""
    args = parse_args()
    export_grid(
        grid_path=args.grid,
        output_dir=args.output_dir,
        source_uri=args.source_uri,
        layer_name=args.layer_name,
        minzoom=args.minzoom,
        maxzoom=args.maxzoom,
    )


if __name__ == "__main__":
    main()
