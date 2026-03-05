"""Extract environmental variables to H3 grid."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from riskscape.config import cfg, paths


def load_lookup(dataset):
    """Load pixel → H3 lookup."""

    lookup_dir = paths.get("lookups", paths["data"] / "lookups")

    file = lookup_dir / f"{dataset}_lookup.parquet"

    df = pd.read_parquet(file)

    pixels = df["pixel"].values

    codes, h3 = pd.factorize(df["h3"])

    h3 = pd.Series(h3).apply(lambda x: int(x, 16)).astype("uint64").values

    return pixels, codes, h3


def open_dataset(dataset):
    """Open dataset."""

    dataset_dir = paths["raw"] / dataset

    files = sorted(dataset_dir.glob("*.nc"))

    if len(files) == 1:
        return xr.open_dataset(files[0])

    return xr.open_mfdataset(files, combine="by_coords")


def aggregate(values, pixels, codes):
    """Aggregate raster values to H3."""

    vals = values[pixels]

    sums = np.bincount(codes, weights=vals)

    counts = np.bincount(codes)

    means = sums / counts

    return means.astype("float32")


def align_to_grid(values, lookup_h3, grid_h3):
    """Align lookup results to grid."""

    out = np.full(len(grid_h3), np.nan, dtype="float32")

    idx = pd.Index(grid_h3).get_indexer(lookup_h3)

    valid = idx >= 0

    out[idx[valid]] = values[valid]

    return out


def main():
    """Run extraction."""

    datasets = list(cfg["layer1"]["variables"])

    # load master H3 grid
    resolution = cfg["grid"]["resolution"]
    region_name = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region_name}.geojson"

    grid = gpd.read_file(grid_file, columns=["id"])

    grid_h3 = grid["id"].apply(lambda x: int(x, 16)).astype("uint64").values

    # open datasets
    dss = {d: open_dataset(d) for d in datasets}

    lookups = {d: load_lookup(d) for d in datasets}

    times = pd.to_datetime(dss[datasets[0]].time.values)

    out_root = paths.get("layer1", paths["data"] / "layer1")

    out_root.mkdir(parents=True, exist_ok=True)

    current_year = None
    year_rows = []

    for t in times:

        date = pd.Timestamp(t)

        year = date.year

        # if year changes → write previous year
        if current_year is not None and year != current_year:

            df_year = pd.concat(year_rows, ignore_index=True)

            out_file = out_root / f"year={current_year}.parquet"

            df_year.to_parquet(
                out_file,
                compression="zstd",
                index=False,
            )

            print("saved", out_file)

            year_rows = []

        current_year = year

        df_day = pd.DataFrame({"h3": grid_h3})

        for dataset in datasets:

            ds = dss[dataset]

            var = cfg["datasets"][dataset]["variable"]

            da = ds[var].sel(time=np.datetime64(date), method="nearest")

            values = da.values.reshape(-1)

            pixels, codes, lookup_h3 = lookups[dataset]

            means = aggregate(values, pixels, codes)

            aligned = align_to_grid(means, lookup_h3, grid_h3)

            df_day[dataset] = aligned.astype("float32")

        df_day["date"] = date

        df_day = df_day[["date", "h3"] + datasets]

        year_rows.append(df_day)

        print("processed", date.date())

    # save final year
    if year_rows:

        df_year = pd.concat(year_rows, ignore_index=True)

        out_file = out_root / f"year={current_year}.parquet"

        df_year.to_parquet(
            out_file,
            compression="zstd",
            index=False,
        )

        print("saved", out_file)

    print("Extraction finished")


if __name__ == "__main__":
    main()