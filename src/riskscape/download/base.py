from pathlib import Path
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor

from riskscape.config import cfg, paths


def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def download_file(url, outfile):

    if outfile.exists():
        return

    r = requests.get(url)

    if r.status_code == 200:
        with open(outfile, "wb") as f:
            f.write(r.content)
    else:
        print("Failed:", url)


def run_downloader(dataset_name, build_url, build_filename):

    start = datetime.date.fromisoformat(cfg["time"]["start"])
    end = datetime.date.fromisoformat(cfg["time"]["end"])

    raw_dir = paths["raw"] / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    for date in daterange(start, end):

        filename = build_filename(date)
        url = build_url(date)

        outfile = raw_dir / filename

        tasks.append((url, outfile))

    # parallel downloads
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda t: download_file(*t), tasks)