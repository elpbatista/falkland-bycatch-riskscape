"""Download datasets defined in config."""

import argparse
import shutil
from pathlib import Path

from riskscape.config import cfg
from riskscape.providers import get_provider


def main():

    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing dataset folders before downloading",
    )

    args = parser.parse_args()

    datasets = cfg["datasets"]
    raw_dir = Path(cfg["paths"]["raw"])

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Optional cleaning step
    if args.clean:

        print("Cleaning dataset folders")

        for name in datasets:

            dataset_dir = raw_dir / name

            if dataset_dir.exists():
                print("Removing:", dataset_dir)
                shutil.rmtree(dataset_dir)

    # Download datasets
    for name, ds in datasets.items():

        provider = get_provider(ds["provider"])

        dataset_dir = raw_dir / name

        print("Downloading:", name)

        provider.download(ds, dataset_dir)


if __name__ == "__main__":
    main()