import datetime

from riskscape.config import cfg
from riskscape.download.base import run_downloader


BASE_URL = cfg["datasets"]["sst"]["base_url"]


def mur_filename(date):

    return (
        f"{date:%Y%m%d}090000-JPL-L4_GHRSST-"
        "SSTfnd-MUR-GLOB-v04.1-fv04.1.nc"
    )


def mur_url(date):

    year = date.strftime("%Y")
    doy = date.strftime("%j")

    return f"{BASE_URL}/{year}/{doy}/{mur_filename(date)}"


def download_sst():

    run_downloader(
        dataset_name="sst",
        build_url=mur_url,
        build_filename=mur_filename
    )