"""Build pixel -> H3 lookup tables using pixel footprint polyfill."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import h3

from riskscape.config import cfg


def detect_coords(ds):
    lat = next((c for c in ("lat", "latitude") if c in ds.coords), None)
    lon = next((c for c in ("lon", "longitude") if c in ds.coords), None)

    if lat is None or lon is None:
        raise KeyError("lat/lon coordinates not found")

    return lat, lon


def open_reference_raster(dataset_name):

    raw_dir = Path(cfg["paths"]["raw"])
    dataset_dir = raw_dir / dataset_name

    files = sorted(dataset_dir.glob("*.nc"))

    if not files:
        raise RuntimeError(f"No raster files found for {dataset_name}")

    return xr.open_dataset(files[0])


def pixel_size(coords):

    diffs = np.diff(coords)

    return float(np.abs(diffs).mean())


def pixel_polygon(lat, lon, dlat, dlon):

    lat_min = lat - dlat / 2
    lat_max = lat + dlat / 2
    lon_min = lon - dlon / 2
    lon_max = lon + dlon / 2

    return {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_min, lat_max],
            [lon_max, lat_max],
            [lon_max, lat_min],
            [lon_min, lat_min],
        ]]
    }


def build_lookup(dataset_name):

    print("Building lookup:", dataset_name)

    ds = open_reference_raster(dataset_name)

    lat_name, lon_name = detect_coords(ds)

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ds.close()

    dlat = pixel_size(lats)
    dlon = pixel_size(lons)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    resolution = int(cfg["grid"]["resolution"])

    rows = []

    for pixel_id, (lat, lon) in enumerate(zip(lat_flat, lon_flat)):

        poly = pixel_polygon(lat, lon, dlat, dlon)

        hexes = h3.geo_to_cells(poly, resolution)

        for h in hexes:
            rows.append((pixel_id, h))

    lookup = pd.DataFrame(rows, columns=["pixel", "h3"])

    lookup_dir = Path(cfg["paths"]["data"]) / "lookups"
    lookup_dir.mkdir(parents=True, exist_ok=True)

    out_file = lookup_dir / f"{dataset_name}_lookup.parquet"

    lookup.to_parquet(out_file, index=False)

    print("pixels:", len(lat_flat))
    print("rows in lookup:", len(lookup))
    print("unique H3 cells:", lookup["h3"].nunique())
    print("saved:", out_file)


def main():

    datasets = cfg["datasets"]

    for name in datasets.keys():
        build_lookup(name)


if __name__ == "__main__":
    main()