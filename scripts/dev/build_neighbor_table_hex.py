import pandas as pd
from pathlib import Path
import h3

from riskscape.config import cfg, paths


def build_neighbor_table():

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    print("Loading grid:", grid_file)
    grid = pd.read_parquet(grid_path)

    h3_cells = grid["h3_index"].astype(str).tolist()
    h3_set = set(h3_cells)

    rows = []

    for h in h3_cells:

        neighbors = list(h3.grid_disk(h, 1))
        neighbors.remove(h)

        neighbors = [n for n in neighbors if n in h3_set]
        neighbors += [None] * (6 - len(neighbors))

        row = {"h3": h}

        for i, n in enumerate(neighbors):
            row[f"n{i+1}"] = n

        rows.append(row)

    neighbor_df = pd.DataFrame(rows)

    output_dir = Path(paths["processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "h3_neighbors.parquet"

    neighbor_df.to_parquet(output_path, index=False)

    print("Saved:", output_path)
    print("Total cells:", len(neighbor_df))
    print(neighbor_df.head())

if __name__ == "__main__":
    build_neighbor_table()