"""
PO.DAAC MUR SST downloader.

Downloads daily global files via HTTPS (Earthdata auth) and crops locally
to the buffered study area from config.
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


def _buffered_bbox_from_cfg() -> tuple[float, float, float, float]:
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


def _dates(start: str, end: str):
    d0 = datetime.fromisoformat(start).date()
    d1 = datetime.fromisoformat(end).date()

    day = d0
    while day <= d1:
        yield day
        day += timedelta(days=1)


def _mur_url(day) -> str:
    return (
        f"{BASE_URL}/"
        f"{day:%Y%m%d}090000"
        "-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
    )


def _curl_download(url: str, out_file: Path) -> None:
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


def _crop_file(in_file: Path, out_file: Path, variable: str) -> None:
    xmin, ymin, xmax, ymax = _buffered_bbox_from_cfg()

    ds = xr.open_dataset(in_file)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        ds.close()
        raise KeyError("Expected 'lat' and 'lon' coordinates in dataset.")

    cropped = ds[[variable]].sel(
        lon=slice(xmin, xmax),
        lat=slice(ymin, ymax),
    )

    cropped.to_netcdf(out_file)
    ds.close()


def _download_and_crop_day(day, variable: str, out_dir: Path, tmp_dir: Path) -> None:
    out_file = out_dir / f"sst_{day:%Y%m%d}.nc"
    if out_file.exists():
        return

    tmp_file = tmp_dir / f"mur_{day:%Y%m%d}.nc"
    url = _mur_url(day)

    print(f"Downloading SST {day.isoformat()}")

    _curl_download(url, tmp_file)
    _crop_file(tmp_file, out_file, variable)

    try:
        tmp_file.unlink()
    except FileNotFoundError:
        pass


def download(dataset_cfg, dataset_dir: Path) -> None:
    """Download SST dataset."""

    variable = dataset_cfg["variable"]

    dataset_dir = Path(dataset_dir)
    tmp_dir = dataset_dir / "_tmp_"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    workers = int(cfg.get("downloads", {}).get("workers", 4))
    days = list(_dates(start, end))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_download_and_crop_day, day, variable, dataset_dir, tmp_dir)
            for day in days
        ]

        for fut in futures:
            fut.result()