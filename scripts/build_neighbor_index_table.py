import pandas as pd
from pathlib import Path

from riskscape.config import cfg, paths


def build_index_table():
    """Build neighbor index table from uint64 grid and neighbor table."""

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}_uint64.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    grid = pd.read_parquet(grid_path)

    if "h3" not in grid.columns:
        raise ValueError("Expected column 'h3' not found in grid")

    grid["h3"] = grid["h3"].astype("uint64")
    grid = grid.sort_values("h3").reset_index(drop=True)

    index_map = {int(h): i for i, h in enumerate(grid["h3"].tolist())}

    neighbor_path = Path(paths["processed"]) / "h3_neighbors.parquet"
    if not neighbor_path.exists():
        raise FileNotFoundError(f"Neighbor table not found: {neighbor_path}")

    neighbors = pd.read_parquet(neighbor_path)

    cols = ["h3"] + [f"n{i}" for i in range(1, 7)]
    for col in cols:
        neighbors[col] = neighbors[col].astype("UInt64")

    rows = []

    for row in neighbors.itertuples(index=False):
        base_idx = index_map[int(row.h3)]

        new_row = {"idx": base_idx}

        for i in range(1, 7):
            n = getattr(row, f"n{i}")

            if pd.isna(n):
                new_row[f"n{i}"] = -1
            else:
                new_row[f"n{i}"] = index_map[int(n)]

        rows.append(new_row)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    output_path = Path(paths["processed"]) / "h3_neighbor_index.parquet"
    df.to_parquet(output_path, index=False)

    print("Saved indexed neighbor table:", output_path)
    print("Total rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    build_index_table()