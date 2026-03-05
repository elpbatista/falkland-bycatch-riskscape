"""Extract Layer 1 daily H3 features using lookup tables (optimized)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from riskscape.config import cfg, paths


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _date_range(start: str, end: str):

    d0 = _parse_date(start)
    d1 = _parse_date(end)

    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _detect_coords(ds: xr.Dataset):

    lat_candidates = ("lat", "latitude")
    lon_candidates = ("lon", "longitude")

    lat = next((c for c in lat_candidates if c in ds.coords), None)
    lon = next((c for c in lon_candidates if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError("lat/lon coords not found")

    return lat, lon


def _open_dataset(dataset_dir: Path):

    files = sorted(dataset_dir.glob("*.nc"))

    if len(files) == 1:
        return xr.open_dataset(files[0])

    return xr.open_mfdataset(files, combine="by_coords")


def _select_day(ds: xr.Dataset, day: date):

    if "time" not in ds.coords:
        return ds

    day0 = np.datetime64(day.isoformat())
    day1 = np.datetime64((day + timedelta(days=1)).isoformat())

    subset = ds.sel(time=slice(day0, day1))

    if subset.sizes.get("time", 0) == 0:
        raise KeyError

    if subset.sizes.get("time", 1) > 1:
        subset = subset.isel(time=0)

    return subset


def _load_lookup(dataset_name: str):

    lookup_dir = paths.get("lookups", paths["data"] / "lookups")
    lookup_path = lookup_dir / f"{dataset_name}_lookup.parquet"

    lookup = pd.read_parquet(lookup_path)

    # Convert H3 ids to categorical codes
    lookup["h3"] = lookup["h3"].astype("category")

    h3_codes = lookup["h3"].cat.codes.values
    h3_index = lookup["h3"].cat.categories

    return h3_codes, h3_index


def _extract_mean(values, h3_codes, n_cells):

    sums = np.zeros(n_cells, dtype="float64")
    counts = np.zeros(n_cells, dtype="int64")

    valid = np.isfinite(values)

    np.add.at(sums, h3_codes[valid], values[valid])
    np.add.at(counts, h3_codes[valid], 1)

    means = sums / counts
    means[counts == 0] = np.nan

    return means


def _extract_centroid(values, h3_codes, n_cells):

    result = np.full(n_cells, np.nan, dtype="float64")

    seen = np.zeros(n_cells, dtype=bool)

    for i, v in enumerate(values):

        if not np.isfinite(v):
            continue

        cell = h3_codes[i]

        if not seen[cell]:
            result[cell] = v
            seen[cell] = True

    return result


def _dataset_dir(dataset_name):

    return paths["raw"] / dataset_name


def main():

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    method = cfg["layer1"]["method"]

    layer_vars = cfg["layer1"]["variables"]

    datasets_cfg = cfg["datasets"]

    all_frames = []

    for dataset_name in layer_vars:

        var = datasets_cfg[dataset_name]["variable"]

        print("Opening:", dataset_name)

        ds = _open_dataset(_dataset_dir(dataset_name))

        h3_codes, h3_index = _load_lookup(dataset_name)

        n_cells = len(h3_index)

        for day in _date_range(start, end):

            try:
                dsd = _select_day(ds, day)
            except KeyError:
                continue

            da = dsd[var]

            values = da.values.reshape(-1)

            if method == "mean":
                agg = _extract_mean(values, h3_codes, n_cells)

            elif method == "centroid":
                agg = _extract_centroid(values, h3_codes, n_cells)

            else:
                raise ValueError("Unknown extraction method")

            frame = pd.DataFrame(
                {
                    "date": day.isoformat(),
                    "h3": h3_index,
                    dataset_name: agg.astype("float32"),
                }
            )

            all_frames.append(frame)

            print(dataset_name, day, "done")

        ds.close()

    df = all_frames[0]

    for other in all_frames[1:]:
        df = df.merge(other, on=["date", "h3"], how="outer")

    out_dir = paths["processed"] / "layer1"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "layer1_h3_daily.parquet"

    df.to_parquet(out_file, index=False)

    print("Saved:", out_file)


if __name__ == "__main__":
    main()