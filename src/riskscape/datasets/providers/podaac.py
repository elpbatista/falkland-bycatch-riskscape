"""
PO.DAAC MUR SST downloader.

Download daily SST files and crop them to the buffered region.
"""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from riskscape.config import cfg


BASE_URL = (
    "https://archive.podaac.earthdata.nasa.gov/"
    "podaac-ops-cumulus-protected/"
    "MUR-JPL-L4-GLOB-v4.1"
)


def buffered_bbox():
    """Return buffered bounding box from config."""

    bbox = cfg["region"]["bbox"]
    buffer_km = float(cfg["region"]["buffer_km"])

    xmin = float(bbox["xmin"])
    ymin = float(bbox["ymin"])
    xmax = float(bbox["xmax"])
    ymax = float(bbox["ymax"])

    mid_lat = (ymin + ymax) / 2.0

    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * math.cos(math.radians(mid_lat)))

    return xmin - dlon, ymin - dlat, xmax + dlon, ymax + dlat


def dates(start, end):
    """Generate dates between start and end."""

    d0 = datetime.fromisoformat(start).date()
    d1 = datetime.fromisoformat(end).date()

    day = d0
    while day <= d1:
        yield day
        day += timedelta(days=1)


def mur_url(day):
    """Build MUR file URL."""

    return (
        f"{BASE_URL}/"
        f"{day:%Y%m%d}090000"
        "-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
    )


def curl_download(url, out_file):
    """Download using curl with Earthdata authentication."""

    cookie_file = Path.home() / ".urs_cookies"

    cmd = [
        "curl",
        "-L",
        "-n",
        "-c",
        str(cookie_file),
        "-b",
        str(cookie_file),
        "-o",
        str(out_file),
        url,
    ]

    subprocess.run(cmd, check=True)


def crop_file(in_file, out_file, variable):
    """Crop dataset to bounding box."""

    xmin, ymin, xmax, ymax = buffered_bbox()

    ds = xr.open_dataset(in_file)

    cropped = ds[[variable]].sel(
        lon=slice(xmin, xmax),
        lat=slice(ymin, ymax),
    )

    cropped.to_netcdf(out_file)

    ds.close()


def download_day(day, variable, out_dir, tmp_dir):
    """Download and crop a single day."""

    final_file = out_dir / f"sst_{day:%Y%m%d}.nc"

    if final_file.exists():
        return

    tmp_file = tmp_dir / f"mur_{day:%Y%m%d}.nc"

    print(f"Downloading SST {day.isoformat()}")

    url = mur_url(day)

    curl_download(url, tmp_file)

    crop_file(tmp_file, final_file, variable)

    try:
        tmp_file.unlink()
    except FileNotFoundError:
        pass


def download(dataset_cfg, dataset_dir):
    """Download SST dataset."""

    variable = dataset_cfg["variable"]

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = dataset_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    workers = int(cfg.get("downloads", {}).get("workers", 4))

    days = list(dates(start, end))

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = [
            executor.submit(
                download_day,
                day,
                variable,
                dataset_dir,
                tmp_dir,
            )
            for day in days
        ]

        for f in futures:
            f.result()