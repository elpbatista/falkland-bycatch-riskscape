import pandas as pd
from pathlib import Path
import h3

from riskscape.config import cfg, paths


def build_neighbor_table():
    """Build H3 neighbor table from uint64 grid without precision loss."""

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}_uint64.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    print("Loading grid:", grid_file)
    grid = pd.read_parquet(grid_path)

    if "h3" not in grid.columns:
        raise ValueError("Expected column 'h3' not found in grid")

    h3_cells = grid["h3"].astype("uint64").tolist()
    h3_set = set(int(h) for h in h3_cells)

    h3_values = []
    neighbor_cols = {f"n{i}": [] for i in range(1, 7)}

    for h in h3_cells:
        h_int = int(h)
        h_hex = h3.int_to_str(h_int)

        neighbors = list(h3.grid_disk(h_hex, 1))
        neighbors.remove(h_hex)

        neighbors = [h3.str_to_int(n) for n in neighbors]
        neighbors = [n for n in neighbors if n in h3_set]
        neighbors += [pd.NA] * (6 - len(neighbors))

        h3_values.append(h_int)

        for i, n in enumerate(neighbors, start=1):
            neighbor_cols[f"n{i}"].append(n)

    neighbor_df = pd.DataFrame({
        "h3": pd.array(h3_values, dtype="UInt64"),
        **{
            col: pd.array(values, dtype="UInt64")
            for col, values in neighbor_cols.items()
        },
    })

    output_dir = Path(paths["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "h3_neighbors.parquet"
    neighbor_df.to_parquet(output_path, index=False)

    print("Saved:", output_path)
    print("Total cells:", len(neighbor_df))
    print(neighbor_df.head())


if __name__ == "__main__":
    build_neighbor_table()