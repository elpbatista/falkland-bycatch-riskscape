"""Download datasets defined in config.yaml."""

import importlib
from pathlib import Path

from riskscape.config import cfg
from riskscape.datasets.catalog import get_datasets


def main():

    datasets = get_datasets()

    raw_root = Path(cfg["paths"]["raw"])

    for name, ds in datasets.items():

        provider_name = ds["provider"]

        provider = importlib.import_module(
            f"riskscape.datasets.providers.{provider_name}"
        )

        raw_dir = raw_root / name
        raw_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading:", name)

        provider.download(ds, raw_dir)


if __name__ == "__main__":
    main()