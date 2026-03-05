"""Download datasets defined in config."""

from pathlib import Path
import shutil

from riskscape.config import cfg
from riskscape.datasets.providers import get_provider


def main():

    datasets = cfg["datasets"]
    raw_dir = Path(cfg["paths"]["raw"])

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing dataset folders
    for name in datasets:

        dataset_dir = raw_dir / name

        if dataset_dir.exists():
            print("Clearing:", dataset_dir)
            shutil.rmtree(dataset_dir)

    # Download datasets
    for name, ds in datasets.items():

        provider = get_provider(ds["provider"])

        dataset_dir = raw_dir / name

        print("Downloading:", name)

        provider.download(ds, dataset_dir)


if __name__ == "__main__":
    main()

    