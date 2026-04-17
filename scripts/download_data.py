"""Download datasets defined in config."""

import argparse
import shutil
from pathlib import Path

from riskscape.config import cfg
from riskscape.providers import get_provider


def main():

    parser = argparse.ArgumentParser(
        description="Download project datasets"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing dataset folders before downloading",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Download only a specific dataset (e.g. wind)",
    )

    args = parser.parse_args()

    datasets = cfg["datasets"]
    raw_dir = Path(cfg["paths"]["raw"])

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Optional cleaning step
    if args.clean:

        print("Cleaning dataset folders")

        targets = (
            [args.dataset]
            if args.dataset
            else datasets.keys()
        )

        for name in targets:

            dataset_dir = raw_dir / name

            if dataset_dir.exists():
                print("Removing:", dataset_dir)
                shutil.rmtree(dataset_dir)

    # Download datasets
    for name, ds in datasets.items():

        if args.dataset and name != args.dataset:
            continue

        provider = get_provider(ds["provider"])

        dataset_dir = raw_dir / name

        print("Downloading:", name)

        provider.download(ds, dataset_dir)


if __name__ == "__main__":
    main()