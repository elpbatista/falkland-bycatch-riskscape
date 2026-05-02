"""Build static H3 feature table."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from shapely.ops import nearest_points

from riskscape.config import cfg, paths
from riskscape.grid import load_grid


logger = logging.getLogger(__name__)

GEOD = pyproj.Geod(ellps="WGS84")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def bathymetry_path() -> Path:
    """Return GEBCO bathymetry path."""
    return paths["raw"] / "bathymetry" / "gebco_2026_clipped.nc"


def output_path() -> Path:
    """Return static feature output path."""
    return paths["data"] / "features" / "static" / "static.parquet"


def detect_coords(ds: xr.Dataset) -> tuple[str, str]:
    """Return latitude and longitude coordinate names."""
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError("lat/lon coordinates not found")

    return lat, lon


def load_bathymetry() -> xr.Dataset:
    """Load GEBCO bathymetry dataset."""
    path = bathymetry_path()

    if not path.exists():
        raise FileNotFoundError(f"Bathymetry file not found: {path}")

    return xr.open_dataset(path)


def geodesic_spacing(lat_values: np.ndarray, lon_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return geodesic pixel spacing in x and y directions."""
    if len(lat_values) < 2 or len(lon_values) < 2:
        raise ValueError("Bathymetry grid must have at least two lat/lon values")

    lon0 = float(lon_values[0])
    lon1 = float(lon_values[1])

    _, _, dx = GEOD.inv(
        np.full_like(lat_values, lon0, dtype="float64"),
        lat_values.astype("float64"),
        np.full_like(lat_values, lon1, dtype="float64"),
        lat_values.astype("float64"),
    )

    lat0 = float(lat_values[0])
    lat1 = float(lat_values[1])

    _, _, dy_value = GEOD.inv(
        lon0,
        lat0,
        lon0,
        lat1,
    )

    dy = np.full(len(lat_values), abs(dy_value), dtype="float64")

    return np.abs(dx), dy


def compute_slope_m_per_m(
    elevation: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
) -> np.ndarray:
    """Compute bathymetric slope magnitude in m/m."""
    dx, dy = geodesic_spacing(lat_values, lon_values)

    dz_dlat_idx, dz_dlon_idx = np.gradient(elevation.astype("float64"))

    dz_dy = dz_dlat_idx / dy[:, None]
    dz_dx = dz_dlon_idx / dx[:, None]

    return np.sqrt(dz_dx**2 + dz_dy**2)


def build_depth_slope_table() -> pd.DataFrame:
    """Build H3 depth and slope table."""
    lookup_path = paths["processed"] / "bathymetry_lookup.parquet"

    if not lookup_path.exists():
        raise FileNotFoundError(f"Bathymetry lookup not found: {lookup_path}")

    with load_bathymetry() as ds:
        lat_name, lon_name = detect_coords(ds)

        elevation = ds["elevation"].values
        lat_values = ds[lat_name].values
        lon_values = ds[lon_name].values

    slope = compute_slope_m_per_m(elevation, lat_values, lon_values)

    pixels = pd.DataFrame(
        {
            "pixel": np.arange(elevation.size, dtype=np.uint32),
            "elevation": elevation.ravel().astype("float32"),
            "slope": slope.ravel().astype("float32"),
        }
    )

    lookup = pd.read_parquet(lookup_path)
    lookup["h3"] = lookup["h3"].astype("uint64")
    lookup["pixel"] = lookup["pixel"].astype("uint32")
    lookup["weight"] = lookup["weight"].astype("float32")

    data = lookup.merge(pixels, on="pixel", how="inner")
    data["depth_weighted"] = -data["elevation"] * data["weight"]
    data["slope_weighted"] = data["slope"] * data["weight"]

    out = (
        data.groupby("h3", as_index=False)
        .agg(
            depth_sum=("depth_weighted", "sum"),
            slope_sum=("slope_weighted", "sum"),
            weight_sum=("weight", "sum"),
        )
    )

    out["depth_m"] = out["depth_sum"] / out["weight_sum"]
    out["slope"] = out["slope_sum"] / out["weight_sum"]

    out = out[["h3", "depth_m", "slope"]].copy()
    out["h3"] = out["h3"].astype("uint64")
    out["depth_m"] = out["depth_m"].astype("float32")
    out["slope"] = out["slope"].astype("float32")

    return out


def load_coastline() -> gpd.GeoDataFrame:
    """Load coastline reference layer."""
    path = PROJECT_ROOT / cfg["references"]["coastline"]

    if not path.exists():
        raise FileNotFoundError(f"Coastline file not found: {path}")

    coast = gpd.read_file(path)

    if coast.crs is None:
        coast = coast.set_crs("EPSG:4326")

    return coast.to_crs("EPSG:4326")


def get_grid_centroids(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return H3 centroid coordinates."""
    return grid[["h3", "lon", "lat"]].rename(
        columns={
            "lon": "centroid_lon",
            "lat": "centroid_lat",
        }
    )


def geodesic_distance_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Return geodesic distance in meters."""
    _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(distance)


def build_distance_to_coast_table(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build H3 distance-to-coast table."""
    coast = load_coastline()
    coast_geom = coast.geometry.union_all()

    centroids = get_grid_centroids(grid)

    distances = []

    for row in centroids.itertuples(index=False):
        point = gpd.points_from_xy(
            [row.centroid_lon],
            [row.centroid_lat],
            crs="EPSG:4326",
        )[0]

        _, nearest = nearest_points(point, coast_geom)

        distance = geodesic_distance_m(
            row.centroid_lon,
            row.centroid_lat,
            nearest.x,
            nearest.y,
        )

        distances.append(distance)

    out = centroids[["h3"]].copy()
    out["dist_coast_m"] = np.array(distances, dtype="float32")

    return out


def build_static_features() -> Path:
    """Build static H3 feature table."""
    grid = load_grid(uint64=True)

    logger.info("Building depth and slope features")
    depth_slope = build_depth_slope_table()

    logger.info("Building distance-to-coast feature")
    dist_coast = build_distance_to_coast_table(grid)

    out = (
        grid[["h3"]]
        .merge(depth_slope, on="h3", how="left")
        .merge(dist_coast, on="h3", how="left")
    )

    out["h3"] = out["h3"].astype("uint64")
    out["depth_m"] = out["depth_m"].astype("float32")
    out["slope"] = out["slope"].astype("float32")
    out["dist_coast_m"] = out["dist_coast_m"].astype("float32")

    path = output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(path, index=False, compression="zstd")

    logger.info("Saved static features: %s", path)
    logger.info("Rows: %d", len(out))

    print(out.head())

    return path