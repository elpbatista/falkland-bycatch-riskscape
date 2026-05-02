"""Download datasets defined in config.

Usage examples:
    python scripts/download_data.py
    python scripts/download_data.py --dataset wind chl sst
    python scripts/download_data.py --dataset wind
    python scripts/download_data.py --clean --dataset wind chl
    python scripts/download_data.py --clean
    python scripts/download_data.py --verbose
    python scripts/download_data.py --dataset wind chl -v
    python scripts/download_data.py --clean --dataset wind chl sst --verbose

Logs are written to the project root `logs/` directory, using a
stage-specific log file.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from riskscape.config import cfg, paths
from riskscape.downloads.providers.loader import get_provider
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context



logger = logging.getLogger(__name__)


def validate_dataset_config(datasets: Dict) -> bool:
    """Validate that dataset configuration is valid."""
    if not datasets:
        logger.error("No datasets defined in configuration")
        return False
    return True


def validate_datasets_exist(names: List[str], datasets: Dict) -> bool:
    """Validate that specific datasets exist in config."""
    invalid_names = [name for name in names if name not in datasets]

    if invalid_names:
        logger.error("Datasets not found: %s", ", ".join(invalid_names))
        logger.info("Available datasets: %s", ", ".join(datasets.keys()))
        return False

    return True


def validate_provider(name: str, provider_name: str) -> bool:
    """Validate that a provider can be loaded."""
    if not provider_name:
        logger.error("Dataset '%s' does not define a provider", name)
        return False

    try:
        get_provider(provider_name)
        return True
    except Exception as exc:
        logger.error(
            "Invalid provider '%s' for dataset '%s': %s",
            provider_name,
            name,
            exc,
        )
        return False


def clean_dataset_folder(dataset_dir: Path) -> bool:
    """Remove a dataset directory if it exists."""
    try:
        if dataset_dir.exists():
            logger.debug("Removing: %s", dataset_dir)
            shutil.rmtree(dataset_dir)
        return True
    except Exception as exc:
        logger.error("Failed to remove %s: %s", dataset_dir, exc)
        return False


def download_dataset(name: str, ds: Dict, dataset_dir: Path) -> bool:
    """Download a single dataset."""
    try:
        provider = get_provider(ds["provider"])
        logger.info("Downloading: %s", name)
        provider.download(ds, dataset_dir)
        logger.info("Successfully downloaded: %s", name)
        return True
    except Exception as exc:
        logger.error("Failed to download '%s': %s", name, exc)
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download project datasets")

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing dataset folders before downloading",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help=(
            "Download specific datasets (e.g. wind chl sst). "
            "If not specified, downloads all."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(stage="download_data", verbose=args.verbose)
    setup_pipeline_logging(verbose=args.verbose)

    with stage_context("download_data"):
        datasets = cfg.get("datasets")
        if not validate_dataset_config(datasets):
            return 1

        if args.dataset and not validate_datasets_exist(args.dataset, datasets):
            return 1

        raw_dir = paths["raw"]
        raw_dir.mkdir(parents=True, exist_ok=True)

        target_datasets = args.dataset if args.dataset else list(datasets.keys())

        if args.clean:
            logger.info("Cleaning dataset folders")
            clean_count = 0

            for name in target_datasets:
                dataset_dir = raw_dir / name
                if clean_dataset_folder(dataset_dir):
                    clean_count += 1

            logger.info(
                "Cleaned %d/%d dataset folders",
                clean_count,
                len(target_datasets),
            )

        download_count = 0
        failed_datasets: List[str] = []

        for name in target_datasets:
            ds = datasets[name]

            if not validate_provider(name, ds.get("provider")):
                failed_datasets.append(name)
                continue

            dataset_dir = raw_dir / name
            if download_dataset(name, ds, dataset_dir):
                download_count += 1
            else:
                failed_datasets.append(name)

        logger.info(
            "Download summary: %d/%d successful",
            download_count,
            len(target_datasets),
        )

        if failed_datasets:
            logger.warning("Failed datasets: %s", ", ".join(failed_datasets))
            return 1

        return 0


if __name__ == "__main__":
    sys.exit(main())