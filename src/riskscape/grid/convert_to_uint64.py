"""Convert grid H3 ids from hex string to uint64."""

from pathlib import Path

import geopandas as gpd
import h3

from riskscape.config import cfg, paths


def convert_grid_to_uint64():
    """Convert grid H3 ids to uint64 and save a new GeoParquet file."""
    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    input_file = f"h3_res{resolution}_{region_name}.parquet"
    input_path = Path(paths["grids"]) / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Grid not found: {input_path}")

    print("Loading grid:", input_file)
    gdf = gpd.read_parquet(input_path)

    if "h3" not in gdf.columns:
        raise ValueError("Expected column 'h3' not found in grid")

    gdf["h3"] = (
        gdf["h3"]
        .astype(str)
        .map(h3.str_to_int)
        .astype("uint64")
    )

    gdf = gdf.sort_values("h3").reset_index(drop=True)

    output_file = f"h3_res{resolution}_{region_name}_uint64.parquet"
    output_path = Path(paths["grids"]) / output_file

    gdf.to_parquet(output_path, engine="pyarrow", index=False)

    print("Saved:", output_path)
    print("Rows:", len(gdf))
    print(gdf.dtypes)
    print(gdf.head())


if __name__ == "__main__":
    convert_grid_to_uint64()