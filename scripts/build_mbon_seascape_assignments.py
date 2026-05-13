"""Build NOAA/MBON 8-day seascape assignments on the project H3/date grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from shapely.geometry import box

from riskscape.config import paths
from riskscape.grid import load_grid
from riskscape.model.dataset import modeling_root
from riskscape.utils.dates import normalize_date_column


MBON_8DAY_OPENDAP_URL = (
    "https://cwcgom.aoml.noaa.gov/thredds/dodsC/"
    "SEASCAPE_8DAY/SEASCAPES.nc"
)
OUTPUT_TABLE = "mbon_seascapes_8day_area_weighted"
DEFAULT_YEARS = "2022"
DEFAULT_BBOX_PAD_DEG = 0.15
GEOD = pyproj.Geod(ellps="WGS84")


def parse_years(years: str) -> list[int]:
    """Parse a single year, range, or comma-separated year list."""
    parsed: set[int] = set()
    for part in years.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            parsed.update(range(int(start_text), int(end_text) + 1))
        else:
            parsed.add(int(item))

    if not parsed:
        raise ValueError("No years selected")

    return sorted(parsed)


def output_path(table_name: str, year: int) -> Path:
    """Return output partition path for one year."""
    return modeling_root(table_name) / f"year={year}" / "part.parquet"


def feature_grid_path(year: int) -> Path:
    """Return feature-grid partition path for one year."""
    return modeling_root("feature_grid") / f"year={year}" / "part.parquet"


def feature_grid_dates(year: int) -> pd.DatetimeIndex:
    """Return daily feature-grid dates for one year."""
    path = feature_grid_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Feature-grid partition not found: {path}")

    dates = pd.read_parquet(path, columns=["date"])["date"].drop_duplicates()
    dates = pd.to_datetime(dates).sort_values()
    return pd.DatetimeIndex(dates)


def coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """Return the first present coordinate name."""
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"Missing coordinate; tried {candidates}")


def open_mbon_dataset(url: str) -> xr.Dataset:
    """Open the NOAA/MBON seascapes dataset."""
    ds = xr.open_dataset(url, decode_times=True)
    required = {"CLASS", "P", "time"}
    missing = required - set(ds.variables)
    if missing:
        raise KeyError(f"NOAA/MBON dataset missing variables: {sorted(missing)}")
    return ds


def pixel_size(coords: np.ndarray) -> float:
    """Return mean absolute pixel spacing."""
    diffs = np.diff(coords)
    return float(np.abs(diffs).mean())


def build_pixel_gdf(lats: np.ndarray, lons: np.ndarray) -> gpd.GeoDataFrame:
    """Build GeoDataFrame of NOAA/MBON raster pixel polygons."""
    dlat = pixel_size(lats)
    dlon = pixel_size(lons)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    n_lon = len(lons)
    pixel_ids = np.arange(len(lat_flat), dtype=np.uint32)
    geometries = [
        box(
            lon - dlon / 2,
            lat - dlat / 2,
            lon + dlon / 2,
            lat + dlat / 2,
        )
        for lat, lon in zip(lat_flat, lon_flat)
    ]

    return gpd.GeoDataFrame(
        {
            "pixel": pixel_ids,
            "lat_idx": (pixel_ids // n_lon).astype(np.uint32),
            "lon_idx": (pixel_ids % n_lon).astype(np.uint32),
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


def geodesic_area_m2(geom: Any) -> float:
    """Return geodesic area in square meters."""
    if geom.is_empty:
        return 0.0
    area, _ = GEOD.geometry_area_perimeter(geom)
    return abs(area)


def build_area_lookup(
    grid_gdf: gpd.GeoDataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
) -> pd.DataFrame:
    """Build H3-to-MBON-pixel geodesic area weights."""
    pixel_gdf = build_pixel_gdf(lats, lons)
    pixel_sindex = pixel_gdf.sindex
    rows: list[dict[str, Any]] = []

    for h3_value, h3_geom in zip(grid_gdf["h3"], grid_gdf.geometry):
        candidate_idx = list(pixel_sindex.intersection(h3_geom.bounds))
        if not candidate_idx:
            continue

        candidates = pixel_gdf.iloc[candidate_idx]
        overlaps: list[tuple[int, int, int, float]] = []

        for pixel_id, lat_idx, lon_idx, pixel_geom in zip(
            candidates["pixel"],
            candidates["lat_idx"],
            candidates["lon_idx"],
            candidates.geometry,
        ):
            inter = h3_geom.intersection(pixel_geom)
            if inter.is_empty:
                continue

            overlap_m2 = geodesic_area_m2(inter)
            if overlap_m2 <= 0:
                continue

            overlaps.append((int(pixel_id), int(lat_idx), int(lon_idx), overlap_m2))

        if not overlaps:
            continue

        total_overlap = sum(area for _, _, _, area in overlaps)
        for pixel_id, lat_idx, lon_idx, overlap_m2 in overlaps:
            rows.append(
                {
                    "h3": int(h3_value),
                    "pixel": pixel_id,
                    "lat_idx": lat_idx,
                    "lon_idx": lon_idx,
                    "overlap_m2": overlap_m2,
                    "weight": overlap_m2 / total_overlap,
                }
            )

    lookup = pd.DataFrame(rows)
    if lookup.empty:
        raise RuntimeError("MBON H3/pixel area lookup is empty")

    lookup["h3"] = lookup["h3"].astype("uint64")
    lookup["pixel"] = lookup["pixel"].astype("uint32")
    lookup["lat_idx"] = lookup["lat_idx"].astype("uint32")
    lookup["lon_idx"] = lookup["lon_idx"].astype("uint32")
    lookup["overlap_m2"] = lookup["overlap_m2"].astype("float64")
    lookup["weight"] = lookup["weight"].astype("float32")
    return lookup


def subset_dataset(
    ds: xr.Dataset,
    grid: pd.DataFrame,
    dates: pd.DatetimeIndex,
    bbox_pad_deg: float,
) -> xr.Dataset:
    """Subset MBON data to the needed time and spatial envelope."""
    lat_name = coord_name(ds, ("lat", "latitude"))
    lon_name = coord_name(ds, ("lon", "longitude"))

    # Pad the date range so nearest 8-day matching works near year edges.
    start = dates.min() - pd.Timedelta(days=8)
    end = dates.max() + pd.Timedelta(days=8)
    time_values = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    time_mask = (time_values >= start) & (time_values <= end)
    selected_times = time_values[time_mask]
    if selected_times.empty:
        raise ValueError(f"No MBON source dates found around {dates.min()}-{dates.max()}")

    lat_min = float(grid["lat"].min()) - bbox_pad_deg
    lat_max = float(grid["lat"].max()) + bbox_pad_deg
    lon_min = float(grid["lon"].min()) - bbox_pad_deg
    lon_max = float(grid["lon"].max()) + bbox_pad_deg

    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values
    lat_slice = (
        slice(lat_min, lat_max)
        if lat_values[0] <= lat_values[-1]
        else slice(lat_max, lat_min)
    )
    lon_slice = (
        slice(lon_min, lon_max)
        if lon_values[0] <= lon_values[-1]
        else slice(lon_max, lon_min)
    )

    return ds.sel(
        time=selected_times,
        **{
            lat_name: lat_slice,
            lon_name: lon_slice,
        },
    )


def sample_mbon_area_weighted(
    ds: xr.Dataset,
    lookup: pd.DataFrame,
    exclude_zero_class: bool,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Aggregate MBON CLASS/P to H3 by dominant geodesic area class."""
    lat_name = coord_name(ds, ("lat", "latitude"))
    lon_name = coord_name(ds, ("lon", "longitude"))
    ds = ds.transpose("time", lat_name, lon_name).load()
    source_dates = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).normalize()
    rows: list[pd.DataFrame] = []
    lat_idx = lookup["lat_idx"].to_numpy(dtype="uint32")
    lon_idx = lookup["lon_idx"].to_numpy(dtype="uint32")

    for time_idx, source_date in enumerate(source_dates):
        classes = np.nan_to_num(
            ds["CLASS"].isel(time=time_idx).to_numpy(),
            nan=0,
        ).astype("int16")
        probabilities = ds["P"].isel(time=time_idx).to_numpy().astype("float32")

        data = lookup[["h3", "weight"]].copy()
        data["mbon_seascape"] = classes[lat_idx, lon_idx]
        data["pixel_probability"] = probabilities[lat_idx, lon_idx]
        data.loc[data["mbon_seascape"] <= 0, "pixel_probability"] = np.nan
        if exclude_zero_class:
            data = data[data["mbon_seascape"] > 0].copy()

        class_weights = (
            data.groupby(["h3", "mbon_seascape"], as_index=False)
            .agg(class_weight=("weight", "sum"))
        )
        probability_data = data.dropna(subset=["pixel_probability"]).copy()
        if probability_data.empty:
            probabilities_by_class = pd.DataFrame(
                columns=["h3", "mbon_seascape", "mbon_probability"]
            )
        else:
            probability_data["weighted_probability"] = (
                probability_data["pixel_probability"] * probability_data["weight"]
            )
            probabilities_by_class = (
                probability_data.groupby(["h3", "mbon_seascape"], as_index=False)
                .agg(
                    p_sum=("weighted_probability", "sum"),
                    w_sum=("weight", "sum"),
                )
            )
            probabilities_by_class["mbon_probability"] = (
                probabilities_by_class["p_sum"] / probabilities_by_class["w_sum"]
            )
            probabilities_by_class = probabilities_by_class[
                ["h3", "mbon_seascape", "mbon_probability"]
            ]

        ranked = class_weights.sort_values(
            ["h3", "class_weight", "mbon_seascape"],
            ascending=[True, False, True],
        ).drop_duplicates("h3")
        ranked = ranked.merge(
            probabilities_by_class,
            on=["h3", "mbon_seascape"],
            how="left",
        )
        ranked["mbon_source_date"] = source_date
        if ranked.empty:
            continue
        rows.append(
            ranked[
                [
                    "h3",
                    "mbon_source_date",
                    "mbon_seascape",
                    "mbon_probability",
                    "class_weight",
                ]
            ]
        )

    if not rows:
        sampled = pd.DataFrame(
            columns=[
                "h3",
                "mbon_source_date",
                "mbon_seascape",
                "mbon_probability",
                "class_weight",
            ]
        )
    else:
        sampled = pd.concat(rows, ignore_index=True)
    sampled["h3"] = sampled["h3"].astype("uint64")
    sampled["mbon_source_date"] = pd.to_datetime(sampled["mbon_source_date"])
    sampled["mbon_seascape"] = sampled["mbon_seascape"].astype("int16")
    sampled["mbon_probability"] = sampled["mbon_probability"].astype("float32")
    sampled["class_weight"] = sampled["class_weight"].astype("float32")
    return source_dates, sampled


def nearest_source_indices(
    dates: pd.DatetimeIndex,
    source_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """Return nearest MBON source-date index for each project date."""
    source_days = source_dates.to_numpy(dtype="datetime64[D]").astype("int64")
    date_days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    positions = np.searchsorted(source_days, date_days)
    positions = np.clip(positions, 0, len(source_days) - 1)
    previous = np.clip(positions - 1, 0, len(source_days) - 1)

    choose_previous = (
        np.abs(date_days - source_days[previous])
        <= np.abs(date_days - source_days[positions])
    )
    return np.where(choose_previous, previous, positions).astype("int16")


def expand_year_assignments(
    grid: pd.DataFrame,
    dates: pd.DatetimeIndex,
    source_dates: pd.DatetimeIndex,
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    """Expand sampled source-date classes to daily H3/date assignments."""
    nearest = nearest_source_indices(dates, source_dates)
    frames: list[pd.DataFrame] = []

    date_days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    source_days = source_dates.to_numpy(dtype="datetime64[D]").astype("int64")

    for source_idx in np.unique(nearest):
        date_indices = np.flatnonzero(nearest == source_idx)
        n_dates = len(date_indices)
        if n_dates == 0:
            continue

        source_date = source_dates[int(source_idx)]
        source_sample = sampled[
            sampled["mbon_source_date"].eq(source_date)
        ].reset_index(drop=True)
        if source_sample.empty:
            continue
        h3_values = source_sample["h3"].to_numpy(dtype="uint64")
        days_from_source = (
            date_days[date_indices] - source_days[int(source_idx)]
        ).astype("int16")

        frame = pd.DataFrame(
            {
                "h3": np.tile(h3_values, n_dates),
                "date": np.repeat(dates[date_indices].to_numpy(), len(h3_values)),
                "mbon_seascape": np.tile(
                    source_sample["mbon_seascape"].to_numpy(dtype="int16"),
                    n_dates,
                ),
                "mbon_probability": np.tile(
                    source_sample["mbon_probability"].to_numpy(dtype="float32"),
                    n_dates,
                ),
                "mbon_class_weight": np.tile(
                    source_sample["class_weight"].to_numpy(dtype="float32"),
                    n_dates,
                ),
                "mbon_source_date": np.repeat(
                    np.datetime64(source_date.date()),
                    n_dates * len(h3_values),
                ),
                "days_from_mbon_source": np.repeat(
                    days_from_source,
                    len(h3_values),
                ),
            }
        )
        frames.append(frame)

    if not frames:
        out = pd.DataFrame(
            columns=[
                "h3",
                "date",
                "mbon_seascape",
                "mbon_probability",
                "mbon_class_weight",
                "mbon_source_date",
                "days_from_mbon_source",
            ]
        )
    else:
        out = pd.concat(frames, ignore_index=True)
    out = normalize_date_column(out)
    out["mbon_source_date"] = pd.to_datetime(out["mbon_source_date"])
    out["mbon_seascape"] = out["mbon_seascape"].astype("int16")
    out["mbon_probability"] = out["mbon_probability"].astype("float32")
    out["mbon_class_weight"] = out["mbon_class_weight"].astype("float32")
    out["days_from_mbon_source"] = out["days_from_mbon_source"].astype("int16")
    return out.sort_values(["date", "h3"]).reset_index(drop=True)


def write_year(
    year: int,
    table_name: str,
    ds: xr.Dataset,
    grid: gpd.GeoDataFrame,
    bbox_pad_deg: float,
    overwrite: bool,
    exclude_zero_class: bool,
) -> Path:
    """Build and write one yearly MBON assignment partition."""
    out_file = output_path(table_name, year)
    if out_file.exists() and not overwrite:
        print(f"Exists, skipping: {out_file}")
        return out_file

    dates = feature_grid_dates(year)
    subset = subset_dataset(ds, grid, dates, bbox_pad_deg=bbox_pad_deg)
    lat_name = coord_name(subset, ("lat", "latitude"))
    lon_name = coord_name(subset, ("lon", "longitude"))
    lookup = build_area_lookup(
        grid_gdf=grid,
        lats=subset[lat_name].values,
        lons=subset[lon_name].values,
    )
    source_dates, sampled = sample_mbon_area_weighted(
        subset,
        lookup,
        exclude_zero_class=exclude_zero_class,
    )
    out = expand_year_assignments(
        grid=grid,
        dates=dates,
        source_dates=source_dates,
        sampled=sampled,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False, compression="zstd")
    print(f"Saved: {out_file}")
    print(f"Rows: {len(out):,}")
    print(
        "MBON source dates:",
        source_dates.min().date(),
        "to",
        source_dates.max().date(),
    )
    return out_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download/sample NOAA/MBON 8-day seascapes onto the project "
            "H3/date feature grid."
        )
    )
    parser.add_argument("--years", default=DEFAULT_YEARS)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
    parser.add_argument("--url", default=MBON_8DAY_OPENDAP_URL)
    parser.add_argument("--bbox-pad-deg", type=float, default=DEFAULT_BBOX_PAD_DEG)
    parser.add_argument(
        "--exclude-zero-class",
        action="store_true",
        help="Exclude MBON CLASS 0 before assigning dominant H3 classes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run MBON seascape assignment workflow."""
    args = parse_args()
    years = parse_years(args.years)

    grid = load_grid(uint64=True)[["h3", "lat", "lon", "geometry"]].copy()
    grid["h3"] = grid["h3"].astype("uint64")
    ds = open_mbon_dataset(args.url)

    for year in years:
        write_year(
            year=year,
            table_name=args.output_table,
            ds=ds,
            grid=grid,
            bbox_pad_deg=args.bbox_pad_deg,
            overwrite=args.overwrite,
            exclude_zero_class=args.exclude_zero_class,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
