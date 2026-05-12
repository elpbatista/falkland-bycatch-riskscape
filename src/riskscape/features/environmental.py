"""Build yearly H3 environmental feature tables."""

from __future__ import annotations

import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from riskscape.config import cfg, paths
from riskscape.utils.dates import normalize_date_column, utc_day


logger = logging.getLogger(__name__)


VARIABLE_ALIASES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}


def detect_coords(ds):
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError("lat/lon coordinates not found")

    return lat, lon


def detect_time(ds):
    return next((c for c in ("time", "valid_time", "date") if c in ds.coords), None)


def list_raster_files(dataset_name: str) -> list[Path]:
    dataset_dir = paths["raw"] / dataset_name
    files = sorted(dataset_dir.glob("*.nc"))
    files.extend(sorted(dataset_dir.glob("*.zip")))
    return files


def iter_datasets(path: Path):
    if path.suffix == ".nc":
        ds = xr.open_dataset(path)
        try:
            yield ds
        finally:
            ds.close()
        return

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith(".nc"):
                    continue
                with z.open(name) as f:
                    ds = xr.open_dataset(f).load()
                try:
                    yield ds
                finally:
                    ds.close()


def get_config_variables(dataset_name: str):
    ds_cfg = cfg["datasets"][dataset_name]

    if "variables" in ds_cfg:
        return list(ds_cfg["variables"])
    if "variable" in ds_cfg:
        return [ds_cfg["variable"]]

    return None


def get_variables(dataset_name: str, ds: xr.Dataset):
    wanted = get_config_variables(dataset_name)

    if wanted is None:
        return list(ds.data_vars)

    resolved = []

    for name in wanted:
        if name in ds.data_vars:
            resolved.append(name)
            continue

        alias = VARIABLE_ALIASES.get(name)
        if alias and alias in ds.data_vars:
            resolved.append(alias)

    return resolved


def feature_name(dataset_name, variable_name, configured):
    if configured and len(configured) > 1:
        return f"{dataset_name}_{variable_name}"
    return dataset_name


def timestamp_utc_day(value):
    """Return UTC-normalized daily timestamp."""
    return utc_day(value)


def aggregate_slice(values, lookup, col):
    flat = values.ravel()
    data = lookup.copy()

    data[col] = flat[data["pixel"].to_numpy()]
    data = data[np.isfinite(data[col])]

    data["w"] = data[col] * data["weight"]

    grouped = data.groupby("h3", as_index=False).agg(
        w_sum=("w", "sum"),
        w_tot=("weight", "sum"),
    )

    grouped[col] = grouped["w_sum"] / grouped["w_tot"]
    return grouped[["h3", col]]


def aggregate_variable(ds, lookup, dataset_name, variable_name):
    lat, lon = detect_coords(ds)
    time = detect_time(ds)

    da = ds[variable_name]

    if time not in da.dims:
        logger.info("Skipping %s:%s (no time dimension)", dataset_name, variable_name)
        return []

    da = da.transpose(time, lat, lon)

    configured = get_config_variables(dataset_name)
    col = feature_name(dataset_name, variable_name, configured)

    frames = []

    for i, t in enumerate(da[time].values):
        values = da.isel({time: i}).values

        df = aggregate_slice(values, lookup, col)
        df["date"] = timestamp_utc_day(t)

        frames.append(df[["h3", "date", col]])

    return frames


def build_dataset_features(dataset_name):
    lookup_path = paths["processed"] / f"{dataset_name}_lookup.parquet"

    if not lookup_path.exists():
        logger.info("Skipping %s (lookup not found)", dataset_name)
        return {}

    files = list_raster_files(dataset_name)
    if not files:
        logger.info("Skipping %s (no raster files)", dataset_name)
        return {}

    lookup = pd.read_parquet(lookup_path)

    by_year = defaultdict(list)

    for path in files:
        logger.info("Processing %s", path.name)

        for ds in iter_datasets(path):
            vars_ = get_variables(dataset_name, ds)

            if not vars_:
                logger.info("Skipping %s:%s (no matching variables)", dataset_name, path.name)
                continue

            for var in vars_:
                frames = aggregate_variable(ds, lookup, dataset_name, var)

                for df in frames:
                    year = pd.to_datetime(df["date"].iloc[0]).year
                    by_year[year].append(df)

    return by_year


def merge_frames(frames):
    df = pd.concat(frames, ignore_index=True)

    feature_cols = [c for c in df.columns if c not in ("h3", "date")]

    df = (
        df.groupby(["h3", "date"], as_index=False)
        .agg({c: "mean" for c in feature_cols})
        .sort_values(["date", "h3"])
        .reset_index(drop=True)
    )

    df["h3"] = df["h3"].astype("uint64")
    df = normalize_date_column(df)

    for c in feature_cols:
        df[c] = df[c].astype("float32")

    return df


def output_root():
    return paths["data"] / "features" / "environmental"


def write_year(df, year):
    out_dir = output_root() / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "part.parquet"
    df = normalize_date_column(df)
    df.to_parquet(out_file, index=False, compression="zstd")

    return out_file


def build_environmental_features():
    year_frames = defaultdict(list)

    for name in cfg["datasets"].keys():
        dataset_frames = build_dataset_features(name)

        for year, frames in dataset_frames.items():
            year_frames[year].extend(frames)

    outputs = []

    for year, frames in sorted(year_frames.items()):
        if not frames:
            continue

        df = merge_frames(frames)
        out_file = write_year(df, year)

        logger.info("Saved: %s", out_file)
        logger.info("Rows: %d", len(df))

        outputs.append(out_file)

    return outputs
