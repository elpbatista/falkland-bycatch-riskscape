#!/usr/bin/env python3
"""Export fisheries grid squares as a Mapbox-ready reference layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = (
    PROJECT_ROOT
    / "reference"
    / "fisheries_grid_squares"
    / "GridSquares.shp"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "plot_exports" / "dashboard" / "mapbox"


def clean_properties(properties: dict[str, Any], grid_id: int) -> dict[str, Any]:
    """Return minimal fisheries-grid properties."""
    label = properties.get("group")
    if label is None:
        raise ValueError(f"Missing fisheries grid group for grid_id={grid_id}")
    return {
        "grid_id": grid_id,
        "fisheries_grid": str(label),
    }


def write_geojsonl(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write line-delimited GeoJSON features."""
    with path.open("w") as file:
        for grid_id, row in enumerate(gdf.itertuples(index=False)):
            properties = clean_properties(row._asdict(), grid_id)
            feature = {
                "type": "Feature",
                "id": grid_id,
                "properties": properties,
                "geometry": row.geometry.__geo_interface__,
            }
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
                    "id": ["get", "grid_id"],
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
    """Export fisheries grid geometry and MTS recipe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(grid_path)
    if gdf.crs is None:
        raise ValueError(f"Missing CRS: {grid_path}")
    gdf = gdf.to_crs("EPSG:4326")

    geojsonl_path = output_dir / "fisheries_grid_squares.geojsonl"
    recipe_path = output_dir / "fisheries_grid_squares_mts_recipe.json"
    write_geojsonl(gdf, geojsonl_path)
    write_recipe(recipe_path, source_uri, layer_name, minzoom, maxzoom)

    for path in [geojsonl_path, recipe_path]:
        print(f"{path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-uri",
        default="mapbox://tileset-source/{username}/falkland-fisheries-grid-squares",
    )
    parser.add_argument("--layer-name", default="fisheries_grid_squares")
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
