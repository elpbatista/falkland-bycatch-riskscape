"""Download datasets defined in config.

Usage examples:
    # Download all datasets
    python scripts/download_data.py

    # Download specific datasets
    python scripts/download_data.py --dataset wind chl sst
    python scripts/download_data.py --dataset wind

    # Clean and re-download
    python scripts/download_data.py --clean --dataset wind chl
    python scripts/download_data.py --clean  # clean all

    # Enable verbose logging
    python scripts/download_data.py --verbose
    python scripts/download_data.py --dataset wind chl -v

    # Combine all options
    python scripts/download_data.py --clean --dataset wind chl sst --verbose

Logs are written to the project root `logs/` directory, using a stage-specific log file.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from riskscape.config import cfg
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.providers import get_provider


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
        logger.error(f"Datasets not found: {', '.join(invalid_names)}")
        logger.info(f"Available datasets: {', '.join(datasets.keys())}")
        return False
    return True


def validate_provider(name: str, provider_name: str) -> bool:
    """Validate that a provider can be instantiated."""
    try:
        get_provider(provider_name)
        return True
    except Exception as e:
        logger.error(f"Invalid provider '{provider_name}' for dataset '{name}': {e}")
        return False


def clean_dataset_folder(dataset_dir: Path) -> bool:
    """Remove a dataset directory if it exists."""
    try:
        if dataset_dir.exists():
            logger.debug(f"Removing: {dataset_dir}")
            shutil.rmtree(dataset_dir)
            return True
    except Exception as e:
        logger.error(f"Failed to remove {dataset_dir}: {e}")
        return False
    return True


def download_dataset(name: str, ds: Dict, dataset_dir: Path) -> bool:
    """Download a single dataset."""
    try:
        provider = get_provider(ds["provider"])
        logger.info(f"Downloading: {name}")
        provider.download(ds, dataset_dir)
        logger.info(f"Successfully downloaded: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to download '{name}': {e}")
        return False


def main() -> int:
    """Main entry point."""
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
        nargs="+",
        help="Download specific datasets (e.g. wind chl sst). If not specified, downloads all.",
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
        # Validate configuration
        datasets = cfg.get("datasets")
        if not validate_dataset_config(datasets):
            return 1

        # Validate dataset filter if provided
        if args.dataset and not validate_datasets_exist(args.dataset, datasets):
            return 1

        raw_dir = Path(cfg["paths"]["raw"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Determine which datasets to process
        target_datasets = args.dataset if args.dataset else list(datasets.keys())

        # Optional cleaning step
        if args.clean:
            logger.info("Cleaning dataset folders")
            clean_count = 0
            for name in target_datasets:
                dataset_dir = raw_dir / name
                if clean_dataset_folder(dataset_dir):
                    clean_count += 1
            logger.info(f"Cleaned {clean_count}/{len(target_datasets)} dataset folders")

        # Download datasets
        download_count = 0
        failed_datasets: List[str] = []

        for name in target_datasets:
            ds = datasets[name]

            # Validate provider before attempting download
            if not validate_provider(name, ds.get("provider")):
                failed_datasets.append(name)
                continue

            dataset_dir = raw_dir / name
            if download_dataset(name, ds, dataset_dir):
                download_count += 1
            else:
                failed_datasets.append(name)

        # Summary
        logger.info(f"Download summary: {download_count}/{len(target_datasets)} successful")

        if failed_datasets:
            logger.warning(f"Failed datasets: {', '.join(failed_datasets)}")
            return 1

        return 0


if __name__ == "__main__":
    sys.exit(main())