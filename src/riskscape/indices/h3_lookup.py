"""Build H3 -> pixel lookup tables using polygon intersection and geodesic area."""

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from shapely.geometry import box

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


logger = logging.getLogger(__name__)

GEOD = pyproj.Geod(ellps="WGS84")


def detect_coords(ds):
    """Return latitude and longitude coordinate names."""
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError("lat/lon coordinates not found")

    return lat, lon


def open_reference_raster(dataset_name):
    """Open one reference dataset for lookup construction."""
    dataset_dir = paths["raw"] / dataset_name

    files = sorted(dataset_dir.glob("*.nc"))
    if files:
        return xr.open_dataset(files[0])

    zips = sorted(dataset_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            nc_files = [n for n in z.namelist() if n.endswith(".nc")]
            if not nc_files:
                return None
            with z.open(nc_files[0]) as f:
                return xr.open_dataset(f).load()

    return None


def pixel_size(coords):
    """Return mean absolute pixel spacing."""
    diffs = np.diff(coords)
    return float(np.abs(diffs).mean())


def build_pixel_gdf(lats, lons):
    """Build GeoDataFrame of raster pixel polygons."""
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


def geodesic_area_m2(geom):
    """Return geodesic area in square meters."""
    if geom.is_empty:
        return 0.0

    area, _ = GEOD.geometry_area_perimeter(geom)
    return abs(area)


def build_dataset_h3_lookup(dataset_name: str) -> Path | None:
    """Build H3 -> pixel lookup for one dataset."""
    logger.info("Building lookup: %s", dataset_name)

    ds = open_reference_raster(dataset_name)
    if ds is None:
        logger.info("Skipping %s (no raster reference found)", dataset_name)
        return None

    lat_name, lon_name = detect_coords(ds)
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    ds.close()

    grid = load_grid(uint64=True)
    pixel_gdf = build_pixel_gdf(lats, lons)
    pixel_sindex = pixel_gdf.sindex

    rows = []

    for h3_value, h3_geom in zip(grid["h3"], grid.geometry):
        candidate_idx = list(pixel_sindex.intersection(h3_geom.bounds))
        if not candidate_idx:
            continue

        candidates = pixel_gdf.iloc[candidate_idx]
        overlaps = []

        for pixel_id, pixel_geom in zip(candidates["pixel"], candidates.geometry):
            inter = h3_geom.intersection(pixel_geom)
            if inter.is_empty:
                continue

            overlap_m2 = geodesic_area_m2(inter)
            if overlap_m2 <= 0:
                continue

            overlaps.append((int(pixel_id), overlap_m2))

        if not overlaps:
            continue

        total_overlap = sum(a for _, a in overlaps)

        for pixel_id, overlap_m2 in overlaps:
            rows.append(
                {
                    "h3": int(h3_value),
                    "pixel": pixel_id,
                    "overlap_m2": overlap_m2,
                    "weight": overlap_m2 / total_overlap,
                }
            )

    lookup = pd.DataFrame(rows)

    if lookup.empty:
        raise RuntimeError(f"Lookup is empty for dataset: {dataset_name}")

    lookup["h3"] = lookup["h3"].astype("uint64")
    lookup["pixel"] = lookup["pixel"].astype("uint32")
    lookup["overlap_m2"] = lookup["overlap_m2"].astype("float64")
    lookup["weight"] = lookup["weight"].astype("float32")

    out_dir = paths["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{dataset_name}_lookup.parquet"
    lookup.to_parquet(out_file, index=False)

    logger.info("Saved: %s", out_file)
    logger.info("Rows: %d", len(lookup))
    logger.info("Unique H3: %d", lookup["h3"].nunique())
    logger.info("Unique pixels: %d", lookup["pixel"].nunique())

    return out_file


def build_h3_lookup() -> list[Path]:
    """Build H3 lookups for all configured datasets."""
    outputs = []

    for name, dataset_cfg in cfg["datasets"].items():
        if not dataset_cfg.get("build_lookup", False):
            logger.info("Skipping lookup for %s (build_lookup is false)", name)
            continue

        out_file = build_dataset_h3_lookup(name)
        if out_file is not None:
            outputs.append(out_file)

    return outputs
