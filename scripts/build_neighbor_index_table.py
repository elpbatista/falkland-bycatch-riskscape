import pandas as pd
from pathlib import Path

from riskscape.config import cfg, paths


def build_index_table():

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    grid = pd.read_parquet(grid_path)

    # Ensure stable ordering
    grid = grid.sort_values("id").reset_index(drop=True)

    # Map h3 id -> integer index
    index_map = {h: i for i, h in enumerate(grid["id"].astype(str))}

    neighbor_path = Path(paths["processed"]) / "h3_neighbors.parquet"
    neighbors = pd.read_parquet(neighbor_path)

    neighbors["h3"] = neighbors["h3"].astype(str)

    rows = []

    for _, row in neighbors.iterrows():

        base_idx = index_map[row["h3"]]

        new_row = {"idx": base_idx}

        for i in range(1, 7):
            n = row[f"n{i}"]
            if pd.isna(n):
                new_row[f"n{i}"] = -1
            else:
                new_row[f"n{i}"] = index_map[str(n)]

        rows.append(new_row)

    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)

    output_path = Path(paths["processed"]) / "h3_neighbor_index.parquet"
    df.to_parquet(output_path, index=False)

    print("Saved indexed neighbor table:", output_path)
    print("Total rows:", len(df))


if __name__ == "__main__":
    build_index_table()