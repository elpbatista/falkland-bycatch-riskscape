"""GEBCO bathymetry downloader via OPeNDAP."""

from __future__ import annotations

import math
import os
from pathlib import Path

import requests
import xarray as xr

from riskscape.config import cfg


DEFAULT_OPENDAP_BASE_URL = "https://dap.ceda.ac.uk/thredds/dodsC"
TOKEN_API_URL = "https://services.ceda.ac.uk/api/token/create/"


def buffered_bbox():
    """Return bounding box including configured buffer."""

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


def coord_slice(coord, lower, upper):
    """Return a slice matching the coordinate's sort order."""

    first = float(coord.values[0])
    last = float(coord.values[-1])

    if first <= last:
        return slice(lower, upper)

    return slice(upper, lower)


def opendap_url(dataset_cfg):
    """Return the OPeNDAP URL for the configured CEDA archive file."""

    if dataset_cfg.get("url"):
        return dataset_cfg["url"]

    archive_path = dataset_cfg["ceda_archive_path"]
    base_url = dataset_cfg.get("opendap_base_url", DEFAULT_OPENDAP_BASE_URL)

    return f"{base_url.rstrip('/')}/{archive_path.lstrip('/')}"


def ceda_access_token():
    """Return a CEDA archive access token if credentials are configured."""

    token = os.environ.get("CEDA_TOKEN")

    if token:
        return token

    username = os.environ.get("CEDA_USERNAME")
    password = os.environ.get("CEDA_PASSWORD")

    if not username or not password:
        return None

    response = requests.post(
        TOKEN_API_URL,
        auth=(username, password),
        timeout=60,
    )
    response.raise_for_status()

    return response.json()["access_token"]


def crop_opendap_dataset(dataset_cfg):
    """Open GEBCO through OPeNDAP and return a cropped dataset."""

    url = opendap_url(dataset_cfg)
    variable = dataset_cfg.get("variable", "elevation")
    lon_name = dataset_cfg.get("lon_name", "lon")
    lat_name = dataset_cfg.get("lat_name", "lat")

    xmin, ymin, xmax, ymax = buffered_bbox()

    token = ceda_access_token()
    session = requests.Session()

    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    try:
        ds = xr.open_dataset(
            url,
            engine=dataset_cfg.get("engine", "pydap"),
            session=session,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not open the GEBCO dataset through CEDA OPeNDAP. "
            "The CEDA service may be temporarily unavailable, or scripted "
            "access may require CEDA_TOKEN or CEDA_USERNAME/CEDA_PASSWORD "
            "in your local .env file. "
            f"URL: {url}"
        ) from exc

    if variable not in ds:
        available = ", ".join(ds.data_vars)
        ds.close()
        raise ValueError(
            f"Variable '{variable}' not found in GEBCO dataset. "
            f"Available variables: {available}"
        )

    cropped = ds[[variable]].sel(
        {
            lon_name: coord_slice(ds[lon_name], xmin, xmax),
            lat_name: coord_slice(ds[lat_name], ymin, ymax),
        }
    )

    return ds, cropped


def output_filename(dataset_cfg):
    """Return the standard local filename for a cropped GEBCO dataset."""

    product = dataset_cfg.get("product", "gebco").lower()
    return f"{product}.nc"


def download(dataset_cfg, dataset_dir):
    """Download a cropped GEBCO bathymetry subset."""

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_file = dataset_dir / output_filename(dataset_cfg)

    if output_file.exists():
        print("Already exists:", output_file)
        return

    product = dataset_cfg.get("product", "GEBCO")
    print("Downloading GEBCO bathymetry:", product)

    ds, cropped = crop_opendap_dataset(dataset_cfg)

    try:
        cropped.load()
        cropped.to_netcdf(output_file)
    finally:
        cropped.close()
        ds.close()

    print("Saved GEBCO subset:", output_file)
