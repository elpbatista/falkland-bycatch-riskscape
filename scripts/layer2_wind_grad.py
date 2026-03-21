"""Add wind_grad to existing Layer 2 yearly tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from riskscape.config import cfg, paths


def year_range():
    """Return inclusive year range from config."""

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    return range(start_year, end_year + 1)


def grid_h3_order():
    """Return master H3 order consistent with neighbor indices."""

    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = f"h3_res{resolution}_{region_name}.parquet"
    grid_path = Path(paths["grids"]) / grid_file

    grid = pd.read_parquet(grid_path)
    grid = grid.sort_values("id").reset_index(drop=True)

    h3_order = (
        grid["id"].apply(lambda x: int(x, 16)).astype("uint64").to_numpy()
    )

    return h3_order


def load_neighbor_idx():
    """Load indexed ring-1 neighbor table."""

    neighbor_path = Path(paths["processed"]) / "h3_neighbor_index.parquet"

    neighbors = pd.read_parquet(neighbor_path)
    neighbors = neighbors.sort_values("idx").reset_index(drop=True)

    return neighbors[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy(
        dtype=np.int64
    )


def build_matrix(df, value_col, h3_order):
    """Build date x cell matrix for one variable."""

    h3_to_idx = {h: i for i, h in enumerate(h3_order)}

    temp = df[["date", "h3", value_col]].copy()
    temp["idx"] = temp["h3"].map(h3_to_idx)

    matrix = (
        temp.pivot(index="date", columns="idx", values=value_col)
        .sort_index(axis=1)
        .to_numpy()
    )

    dates = pd.to_datetime(
        temp[["date"]].drop_duplicates()["date"].sort_values().to_numpy()
    )

    return dates, matrix


def rms_gradient(matrix, neighbor_idx):
    """Compute vectorized RMS neighbor gradient."""

    safe_idx = neighbor_idx.copy()
    topo_valid = safe_idx >= 0
    safe_idx[~topo_valid] = 0

    neighbors = matrix[:, safe_idx]
    center = matrix[:, :, None]

    diff_sq = (center - neighbors) ** 2

    valid = (
        topo_valid[None, :, :]
        & np.isfinite(center)
        & np.isfinite(neighbors)
    )

    sum_sq = np.where(valid, diff_sq, 0.0).sum(axis=2)
    counts = valid.sum(axis=2)

    out = np.full(sum_sq.shape, np.nan, dtype=np.float64)
    np.divide(sum_sq, counts, out=out, where=counts > 0)

    out = np.sqrt(out)

    return out.astype("float32")


def flatten_matrix(dates, h3_order, matrix, value_name):
    """Flatten date x cell matrix back to long format."""

    temp = pd.DataFrame(matrix, index=dates, columns=h3_order)

    return (
        temp.stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "h3", 0: value_name})
    )


def extract_year(year):
    """Add wind_grad to existing Layer 2 file for one year."""

    print(f"\nProcessing wind_grad for {year}")

    layer2_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"
    if not layer2_path.exists():
        raise FileNotFoundError(f"Layer 2 file not found: {layer2_path}")

    df = pd.read_parquet(layer2_path)

    if "wind" not in df.columns:
        raise KeyError("wind column not found in Layer 2 file")

    if "wind_grad" in df.columns:
        print("Dropping existing wind_grad")
        df = df.drop(columns=["wind_grad"])

    h3_order = grid_h3_order()
    neighbor_idx = load_neighbor_idx()

    dates, wind_mat = build_matrix(df, "wind", h3_order)

    print("Computing wind_grad")
    wind_grad = rms_gradient(wind_mat, neighbor_idx)

    wind_grad_df = flatten_matrix(dates, h3_order, wind_grad, "wind_grad")

    layer2 = df.merge(wind_grad_df, on=["date", "h3"], how="left")
    layer2["wind_grad"] = layer2["wind_grad"].astype("float32")

    layer2.to_parquet(layer2_path, index=False)

    print("Saved:", layer2_path)
    print("Rows:", len(layer2))


def main():
    """Run wind_grad extraction for all years."""

    for year in year_range():
        extract_year(year)


if __name__ == "__main__":
    main()