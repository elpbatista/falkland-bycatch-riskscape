"""Copernicus Climate Data Store (CDS) dataset downloader."""

import math
from pathlib import Path

import cdsapi

from riskscape.config import cfg


def buffered_bbox():
    """Return bounding box including configured buffer."""

    bbox = cfg["region"]["bbox"]
    buffer_km = cfg["region"]["buffer_km"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    mid_lat = (ymin + ymax) / 2

    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * math.cos(math.radians(mid_lat)))

    return xmin - dlon, xmax + dlon, ymin - dlat, ymax + dlat


def download(dataset_cfg, dataset_dir):
    """Download ERA5 daily wind data from CDS."""

    product = dataset_cfg["product"]
    variable_u = dataset_cfg["variable_u"]
    variable_v = dataset_cfg["variable_v"]

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    xmin, xmax, ymin, ymax = buffered_bbox()

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    years = range(
        int(start[:4]),
        int(end[:4]) + 1,
    )

    client = cdsapi.Client()

    for year in years:
        output_file = dataset_dir / f"era5_wind_daily_{year}.nc"

        if output_file.exists():
            print(f"Already exists: {year}")
            continue

        print("Downloading ERA5 daily wind:", year)

        client.retrieve(
            product,
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": str(year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "daily_statistic": "daily_mean",
                "time_zone": "UTC+00:00",
                "area": [
                    ymax,   # north
                    xmin,   # west
                    ymin,   # south
                    xmax,   # east
                ],
                "format": "netcdf",
            },
            str(output_file),
        )