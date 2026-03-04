"""
Utilities for downloading daily environmental datasets.
"""

from pathlib import Path
import datetime
import csv
import requests
from concurrent.futures import ThreadPoolExecutor

from riskscape.config import cfg, paths
from riskscape.download.metadata import write_dataset_metadata


def daterange(start, end):
    """Yield dates between start and end (inclusive)."""
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def load_manifest(manifest_file):
    """Return set of filenames already downloaded."""
    completed = set()

    if manifest_file.exists():
        with open(manifest_file) as f:
            reader = csv.reader(f)
            for row in reader:
                completed.add(row[0])

    return completed


def append_manifest(manifest_file, filename):
    """Append filename to manifest."""
    with open(manifest_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename])


def download_file(url, outfile):
    """Download file if it does not exist."""

    if outfile.exists():
        return True

    try:
        r = requests.get(url, timeout=60)

        if r.status_code == 200:
            with open(outfile, "wb") as f:
                f.write(r.content)
            return True

    except Exception:
        pass

    print("Failed:", url)
    return False


def run_downloader(dataset_name, build_url, build_filename):
    """Download dataset for configured time range."""

    start = datetime.date.fromisoformat(cfg["time"]["start"])
    end = datetime.date.fromisoformat(cfg["time"]["end"])

    raw_dir = paths["raw"] / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = paths["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = logs_dir / f"{dataset_name}_manifest.csv"

    dataset_cfg = cfg["datasets"].get(dataset_name, {})
    write_dataset_metadata(dataset_name, dataset_cfg, raw_dir)

    completed = load_manifest(manifest_file)

    tasks = []

    for date in daterange(start, end):

        filename = build_filename(date)

        if filename in completed:
            continue

        url = build_url(date)
        outfile = raw_dir / filename

        tasks.append((url, outfile, filename))

    total_days = (end - start).days + 1
    remaining = len(tasks)
    already_done = total_days - remaining

    print()
    print(f"Dataset: {dataset_name}")
    print(f"Total days: {total_days}")
    print(f"Already downloaded: {already_done}")
    print(f"Remaining downloads: {remaining}")
    print()

    def worker(task):
        url, outfile, filename = task

        if download_file(url, outfile):
            append_manifest(manifest_file, filename)
            print("Downloaded:", filename)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(worker, tasks)