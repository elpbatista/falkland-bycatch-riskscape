"""Provider loader."""

from . import copernicus
from . import podaac
from . import cds

def get_provider(name):

    if name == "copernicus":
        return copernicus

    if name == "podaac":
        return podaac
    
    if name == "cds":
        return cds

    raise ValueError(f"Unknown provider: {name}")