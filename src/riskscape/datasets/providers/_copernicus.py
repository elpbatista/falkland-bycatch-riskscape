"""Copernicus Marine dataset downloader."""

import math
import subprocess
from pathlib import Path

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


def download(dataset_cfg, output_dir):
    """Download Copernicus dataset."""

    product = dataset_cfg["product"]
    variable = dataset_cfg["variable"]

    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    xmin, xmax, ymin, ymax = buffered_bbox()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading:", product)

    cmd = [
        "copernicusmarine",
        "subset",
        "--dataset-id", product,
        "--variable", variable,
        "--start-datetime", start,
        "--end-datetime", end,
        "--minimum-longitude", str(xmin),
        "--maximum-longitude", str(xmax),
        "--minimum-latitude", str(ymin),
        "--maximum-latitude", str(ymax),
        "--output-directory", str(output_dir),
        "--file-format", "netcdf"
    ]

    subprocess.run(cmd, check=True)