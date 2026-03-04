"""
Download a dataset defined in config.yaml.
"""

import sys

from riskscape.download.base import run_downloader
from riskscape.download.datasets import DATASET_BUILDERS


def main():

    if len(sys.argv) < 2:
        print("Usage: python download_dataset.py <dataset>")
        sys.exit(1)

    dataset_name = sys.argv[1]

    if dataset_name not in DATASET_BUILDERS:
        print("Unknown dataset:", dataset_name)
        sys.exit(1)

    builder = DATASET_BUILDERS[dataset_name]

    run_downloader(
        dataset_name=dataset_name,
        build_url=builder["url"],
        build_filename=builder["filename"]
    )


if __name__ == "__main__":
    main()