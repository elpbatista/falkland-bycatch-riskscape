"""Download fishing effort data from Global Fishing Watch 4Wings API."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import gfwapiclient as gfw
import pandas as pd

from riskscape.config import cfg, paths


def build_buffered_bbox() -> tuple[float, float, float, float]:
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


def build_geojson() -> dict:
    """Build a custom GeoJSON polygon from the buffered bbox."""
    xmin, ymin, xmax, ymax = build_buffered_bbox()

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


async def fetch_fishing_effort(client, payload):
    """Fetch fishing effort report."""
    return await client.fourwings.create_fishing_effort_report(**payload)


def main() -> None:
    """Download and store fishing effort as Parquet."""
    token = cfg["gfw"]["token"]
    if not token:
        raise RuntimeError("GFW token is not set in config.yaml")

    client = gfw.Client(access_token=token)
    geojson = build_geojson()

    payload = {
        "spatial_resolution": "HIGH",
        "temporal_resolution": "DAILY",
        "start_date": cfg["time"]["start"],
        "end_date": cfg["time"]["end"],
        "geojson": geojson,
    }

    result = asyncio.run(fetch_fishing_effort(client, payload))
    df = result.df()

    # --- configurable columns ---
    keep = [
        "date",
        "hours",
        "lat",
        "lon",
        "flag",
        "gear_type",
        # "vessel_id",
        "vessel_type",
        # "detections",
        # "vessel_ids",
        # "entry_timestamp",
        # "exit_timestamp",
        # "first_transmission_date",
        # "last_transmission_date",
        # "imo",
        # "mmsi",
        # "call_sign",
        # "dataset",
        # "report_dataset",
        "ship_name",
    ]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[keep].copy()

    # --- enforce types ---
    df["date"] = pd.to_datetime(df["date"], utc=True).astype("int64")
    df["hours"] = df["hours"].astype("float32")
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")

    # --- output ---
    out_dir = Path(paths["raw"]) / "fishing_effort"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "fishing_effort.parquet"
    df.to_parquet(out_file, index=False)

    print("Saved:", out_file)
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()