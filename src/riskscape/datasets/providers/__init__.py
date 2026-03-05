"""Provider loader."""

from . import copernicus
from . import podaac


def get_provider(name):

    if name == "copernicus":
        return copernicus

    if name == "podaac":
        return podaac

    raise ValueError(f"Unknown provider: {name}")