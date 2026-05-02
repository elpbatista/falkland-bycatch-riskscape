"""Extent utilities for the H3 grid and downloads."""

import math

from riskscape.config import cfg


def get_buffered_bbox() -> tuple[float, float, float, float]:
    """Return buffered bbox from config."""
    bbox = cfg["region"]["bbox"]
    buffer_km = cfg["region"]["buffer_km"]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    mid_lat = (ymin + ymax) / 2.0

    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * math.cos(math.radians(mid_lat)))

    return xmin - dlon, ymin - dlat, xmax + dlon, ymax + dlat


def get_buffered_extent_dict() -> dict:
    """Return buffered bbox as a dict."""
    xmin, ymin, xmax, ymax = get_buffered_bbox()

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def get_buffered_polygon_geojson() -> dict:
    """Return buffered bbox as a GeoJSON polygon."""
    xmin, ymin, xmax, ymax = get_buffered_bbox()

    return {
        "type": "Polygon",
        "coordinates": [[
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]],
    }