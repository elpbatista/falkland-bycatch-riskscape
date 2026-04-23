"""Download fishing effort data (year-partitioned) from GFW 4Wings API."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from turtle import delay

import gfwapiclient as gfw
import pandas as pd

from riskscape.config import cfg, paths


def log(msg: str) -> None:
    """Lightweight logger."""
    print(f"[fishing_effort] {msg}")


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
    """Build polygon from buffered bbox."""
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


def build_year_ranges(start_date: str, end_date: str):
    """Yield yearly ranges."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    for year in range(start.year, end.year + 1):
        y0 = pd.Timestamp(year=year, month=1, day=1)
        y1 = pd.Timestamp(year=year, month=12, day=31)

        s = max(start, y0)
        e = min(end, y1)

        if s <= e:
            yield year, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


async def fetch_with_retry(
    client,
    payload: dict,
    year: int,
    max_attempts: int = 5,
    base_sleep: float = 10.0,
) -> pd.DataFrame:
    """Fetch one yearly fishing-effort report with retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            result = await client.fourwings.create_fishing_effort_report(**payload)
            df = result.df()
            return df

        except Exception as exc:
            is_last = attempt == max_attempts
            wait_s = base_sleep * attempt

            log(f"{year} request failed (attempt {attempt}/{max_attempts}): {exc}")

            if is_last:
                raise

            log(f"{year} retrying in {wait_s:.1f}s")
            await asyncio.sleep(wait_s)

    raise RuntimeError("Retry loop ended unexpectedly")


async def async_main() -> None:
    """Download and store fishing effort as Parquet."""
    token = cfg["gfw"]["token"]
    if not token:
        raise RuntimeError("Missing GFW token in config.yaml")

    client = gfw.Client(access_token=token)
    geojson = build_geojson()

    out_root = Path(paths["raw"]) / "fishing_effort"
    out_root.mkdir(parents=True, exist_ok=True)

    year_ranges = list(build_year_ranges(cfg["time"]["start"], cfg["time"]["end"]))

    keep = [
        "date",
        "hours",
        "lat",
        "lon",
        "flag",
        "gear_type",
        "vessel_id",
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
        # "ship_name",
    ]

    for i, (year, start_date, end_date) in enumerate(year_ranges):
        year_dir = out_root / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)

        out_file = year_dir / "fishing_effort.parquet"

        if out_file.exists():
            log(f"{year} skipped (already exists)")
        else:
            log(f"{year} downloading ({start_date} -> {end_date})")

            payload = {
                "spatial_resolution": "HIGH",
                "temporal_resolution": "DAILY",
                "start_date": start_date,
                "end_date": end_date,
                "geojson": geojson,
            }

            df = await fetch_with_retry(client, payload, year=year)

            if df.empty:
                log(f"{year} empty")
            else:
                missing = [c for c in keep if c not in df.columns]
                if missing:
                    raise ValueError(f"{year} missing expected columns: {missing}")

                df = df[keep].copy()

                df["date"] = pd.to_datetime(df["date"], utc=True).astype("int64")
                df["hours"] = df["hours"].astype("float32")
                df["lat"] = df["lat"].astype("float32")
                df["lon"] = df["lon"].astype("float32")

                df = df.drop_duplicates().reset_index(drop=True)
                df.to_parquet(out_file, index=False)

                log(f"{year} saved ({len(df)} rows)")

        # Wait between yearly requests to avoid concurrent-report / rate-limit issues.
        if i < len(year_ranges) - 1:
            delay = cfg["gfw"].get("request_delay_seconds", 60)
            log(f"waiting {delay}s before next request")
            await asyncio.sleep(delay)

    log("done")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()