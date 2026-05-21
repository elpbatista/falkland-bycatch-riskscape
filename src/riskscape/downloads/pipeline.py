"""Download configured provider-backed datasets."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from riskscape.config import cfg, paths
from riskscape.downloads.providers.loader import get_provider


logger = logging.getLogger(__name__)


def validate_dataset_config(datasets: dict[str, Any] | None) -> bool:
    """Validate that dataset configuration is valid."""
    if not datasets:
        logger.error("No datasets defined in configuration")
        return False
    return True


def validate_datasets_exist(names: list[str], datasets: dict[str, Any]) -> bool:
    """Validate that specific datasets exist in config."""
    invalid_names = [name for name in names if name not in datasets]

    if invalid_names:
        logger.error("Datasets not found: %s", ", ".join(invalid_names))
        logger.info("Available datasets: %s", ", ".join(datasets.keys()))
        return False

    return True


def downloadable_dataset_names(datasets: dict[str, Any]) -> list[str]:
    """Return dataset names that define a download provider."""
    return [name for name, dataset in datasets.items() if dataset.get("provider")]


def validate_provider(name: str, provider_name: str | None) -> bool:
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


def download_dataset(name: str, dataset: dict[str, Any], dataset_dir: Path) -> bool:
    """Download a single dataset."""
    try:
        provider = get_provider(dataset["provider"])
        logger.info("Downloading: %s", name)
        provider.download(dataset, dataset_dir)
        logger.info("Successfully downloaded: %s", name)
        return True
    except Exception as exc:
        logger.error("Failed to download '%s': %s", name, exc)
        return False


def download_configured_datasets(
    dataset_names: list[str] | None = None,
    clean: bool = False,
) -> int:
    """Download configured datasets and return a process-style status code."""
    datasets = cfg.get("datasets")
    if not validate_dataset_config(datasets):
        return 1

    if dataset_names and not validate_datasets_exist(dataset_names, datasets):
        return 1

    raw_dir = paths["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_datasets = (
        dataset_names
        if dataset_names
        else downloadable_dataset_names(datasets)
    )

    if clean:
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
    failed_datasets: list[str] = []

    for name in target_datasets:
        dataset = datasets[name]

        if not validate_provider(name, dataset.get("provider")):
            failed_datasets.append(name)
            continue

        dataset_dir = raw_dir / name
        if download_dataset(name, dataset, dataset_dir):
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
