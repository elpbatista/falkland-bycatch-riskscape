"""PO.DAAC MUR SST downloader."""

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
    """Return buffered bounding box."""

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


def date_range(start, end):
    """Yield dates between start and end."""

    d0 = datetime.fromisoformat(start).date()
    d1 = datetime.fromisoformat(end).date()

    day = d0

    while day <= d1:
        yield day
        day += timedelta(days=1)


def mur_url(day):
    """Return daily dataset URL."""

    return (
        f"{BASE_URL}/"
        f"{day:%Y%m%d}090000"
        "-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
    )


def curl_download(url, out_file):
    """Download file with Earthdata authentication."""

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
    """Crop dataset to buffered bbox."""

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

    out_file = out_dir / f"sst_{day:%Y%m%d}.nc"

    if out_file.exists():
        return

    tmp_file = tmp_dir / f"mur_{day:%Y%m%d}.nc"

    url = mur_url(day)

    print("Downloading SST", day.isoformat())

    curl_download(url, tmp_file)

    crop_file(tmp_file, out_file, variable)

    tmp_file.unlink(missing_ok=True)


def download(dataset_cfg, raw_dir):
    """Download SST dataset."""

    variable = dataset_cfg["variable"]

    raw_dir = Path(raw_dir)
    tmp_dir = raw_dir / "_tmp"

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    days = list(date_range(start, end))

    workers = int(cfg.get("downloads", {}).get("workers", 4))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(download_day, d, variable, raw_dir, tmp_dir)
            for d in days
        ]

        for f in futures:
            f.result()