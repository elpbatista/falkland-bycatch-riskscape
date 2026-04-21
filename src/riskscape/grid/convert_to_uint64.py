"""Convert grid H3 ids from hex string to uint64."""

from pathlib import Path

import h3
import pandas as pd

from riskscape.config import cfg, paths


def convert_grid_to_uint64():
    """Convert grid H3 ids to uint64 and save a new parquet file."""
    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    input_file = f"h3_res{resolution}_{region_name}.parquet"
    input_path = Path(paths["grids"]) / input_file

    if not input_path.exists():
        raise FileNotFoundError(f"Grid not found: {input_path}")

    print("Loading grid:", input_file)
    df = pd.read_parquet(input_path)

    if "h3_index" not in df.columns:
        raise ValueError("Expected column 'h3_index' not found in grid")

    df["h3_index"] = (
        df["h3_index"]
        .astype(str)
        .map(h3.str_to_int)
        .astype("uint64")
    )

    df = df.sort_values("h3_index").reset_index(drop=True)

    output_file = f"h3_res{resolution}_{region_name}_uint64.parquet"
    output_path = Path(paths["grids"]) / output_file

    df.to_parquet(output_path, index=False)

    print("Saved:", output_path)
    print("Rows:", len(df))
    print(df.dtypes)
    print(df.head())


if __name__ == "__main__":
    convert_grid_to_uint64()