"""
Dataset definitions for the downloader.
"""

from riskscape.config import cfg


def build_sst_filename(date):

    return (
        f"{date:%Y%m%d}090000-JPL-L4_GHRSST-"
        "SSTfnd-MUR-GLOB-v04.1-fv04.1.nc"
    )


def build_sst_url(date):

    base_url = cfg["datasets"]["sst"]["base_url"]

    year = date.strftime("%Y")
    doy = date.strftime("%j")

    filename = build_sst_filename(date)

    return f"{base_url}/{year}/{doy}/{filename}"


DATASET_BUILDERS = {

    "sst": {
        "url": build_sst_url,
        "filename": build_sst_filename
    }

}