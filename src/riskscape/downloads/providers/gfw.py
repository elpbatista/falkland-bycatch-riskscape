"""Download fishing effort data (year-partitioned) from GFW 4Wings API."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import gfwapiclient as gfw
import pandas as pd

from riskscape.config import cfg
from riskscape.downloads.common import build_year_ranges
from riskscape.grid import get_buffered_polygon_geojson


logger = logging.getLogger(__name__)


def api_end_date(end_date: str) -> str:
    """Return exclusive API end date."""
    return (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


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
            result = await client.fourwings.create_fishing_effort_report(
                **payload
            )
            return result.df()

        except Exception as exc:
            is_last = attempt == max_attempts
            wait_s = base_sleep * attempt

            logger.warning(
                "%s request failed (attempt %d/%d): %s",
                year,
                attempt,
                max_attempts,
                exc,
            )

            if is_last:
                raise

            logger.info("%s retrying in %.1fs", year, wait_s)
            await asyncio.sleep(wait_s)

    raise RuntimeError("Retry loop ended unexpectedly")


async def _async_download(dataset_dir: Path) -> None:
    """Async download implementation."""
    token = cfg["gfw"]["token"]
    if not token:
        raise RuntimeError("Missing GFW token in config.yaml")

    client = gfw.Client(access_token=token)
    geojson = get_buffered_polygon_geojson()

    dataset_dir.mkdir(parents=True, exist_ok=True)

    year_ranges = list(
        build_year_ranges(cfg["time"]["start"], cfg["time"]["end"])
    )

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
        year_dir = dataset_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)

        out_file = year_dir / "fishing_effort.parquet"

        if out_file.exists():
            logger.info("%s skipped (already exists)", year)
        else:
            request_end_date = api_end_date(end_date)

            logger.info(
                "%s downloading (%s -> %s)",
                year,
                start_date,
                request_end_date,
            )

            payload = {
                "spatial_resolution": "HIGH",
                "temporal_resolution": "DAILY",
                "start_date": start_date,
                "end_date": request_end_date,
                "geojson": geojson,
            }

            df = await fetch_with_retry(client, payload, year=year)

            if df.empty:
                logger.info("%s empty", year)
            else:
                missing = [c for c in keep if c not in df.columns]
                if missing:
                    raise ValueError(
                        f"{year} missing expected columns: {missing}"
                    )

                df = df[keep].copy()

                df["date"] = pd.to_datetime(
                    df["date"],
                    utc=True,
                ).astype("int64")
                df["hours"] = df["hours"].astype("float32")
                df["lat"] = df["lat"].astype("float32")
                df["lon"] = df["lon"].astype("float32")

                df = df.drop_duplicates().reset_index(drop=True)
                df.to_parquet(out_file, index=False)

                logger.info("%s saved (%d rows)", year, len(df))

        if i < len(year_ranges) - 1:
            delay = cfg["gfw"].get("request_delay_seconds", 60)
            logger.info("Waiting %ds before next request", delay)
            await asyncio.sleep(delay)

    logger.info("Download completed")


def download(ds: dict, dataset_dir: Path) -> None:
    """Provider entry point."""
    asyncio.run(_async_download(dataset_dir))