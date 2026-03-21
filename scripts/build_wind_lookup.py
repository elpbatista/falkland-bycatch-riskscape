"""Build ERA5 wind pixel -> H3 lookup using pixel footprint polyfill."""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import h3

from riskscape.config import cfg


def open_reference_wind_dataset():
    """Open first available wind ZIP and return dataset."""

    raw_dir = Path(cfg["paths"]["raw"]) / "wind"
    zips = sorted(raw_dir.glob("*.zip"))

    if not zips:
        raise RuntimeError("No wind ZIP files found")

    zip_path = zips[0]

    with zipfile.ZipFile(zip_path) as z:
        nc_name = [
            n for n in z.namelist()
            if "u_component" in n
        ][0]

        with z.open(nc_name) as f:
            ds = xr.open_dataset(f)

    return ds


def pixel_size(coords):
    """Return mean absolute pixel spacing."""
    diffs = np.diff(coords)
    return float(np.abs(diffs).mean())


def pixel_polygon(lat, lon, dlat, dlon):
    """Return GeoJSON polygon for pixel footprint."""
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


def build_lookup():

    print("Building wind lookup")

    ds = open_reference_wind_dataset()

    lat_name = next(c for c in ("lat", "latitude") if c in ds.coords)
    lon_name = next(c for c in ("lon", "longitude") if c in ds.coords)

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

    out_file = lookup_dir / "wind_lookup.parquet"

    if out_file.exists():
        raise RuntimeError(
            "wind_lookup.parquet already exists. "
            "Delete manually if you want to rebuild."
        )

    lookup.to_parquet(out_file, index=False)

    print("pixels:", len(lat_flat))
    print("rows in lookup:", len(lookup))
    print("unique H3 cells:", lookup["h3"].nunique())
    print("saved:", out_file)


if __name__ == "__main__":
    build_lookup()