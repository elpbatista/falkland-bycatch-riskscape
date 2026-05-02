"""Shared logging helpers for the riskscape package."""

import logging
from pathlib import Path
from typing import Optional

from .file_manager import rotate_log_on_date_change, ensure_logs_dir

PIPELINE_LOGGER_NAME = "riskscape.pipeline"


def _sanitize_stage_name(name: str) -> str:
    return name.strip().replace(" ", "_").lower()


def setup_logging(
    stage: str = "riskscape",
    verbose: bool = False,
    logs_dir: Optional[Path] = None,
) -> Path:
    """Configure console and file logging for the project.

    The log file is written to `logs/<stage>.log` under the project root.
    Existing logs are archived with a timestamped suffix.
    """
    level = logging.DEBUG if verbose else logging.INFO
    stage_name = _sanitize_stage_name(stage)

    logs_dir = ensure_logs_dir(logs_dir)
    log_file = logs_dir / f"{stage_name}.log"
    rotate_log_on_date_change(log_file)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party libraries that emit repeated INFO logs during file output.
    for noisy_logger in (
        "fiona",
        "geopandas",
        "pyogrio",
        "pyproj",
        "urllib3",
        "s3fs",
    ):
        logger = logging.getLogger(noisy_logger)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    return log_file


def setup_pipeline_logging(
    verbose: bool = False,
    logs_dir: Optional[Path] = None,
) -> Path:
    """Configure a general pipeline lifecycle logger."""
    level = logging.DEBUG if verbose else logging.INFO

    logs_dir = ensure_logs_dir(logs_dir)
    log_file = logs_dir / "pipeline.log"
    rotate_log_on_date_change(log_file)

    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    if pipeline_logger.handlers:
        pipeline_logger.handlers.clear()

    pipeline_logger.setLevel(level)
    pipeline_logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    pipeline_logger.addHandler(file_handler)

    return log_file


class stage_context:
    """Context manager for stage lifecycle logging.""" 

    def __init__(self, stage_name: str):
        self.stage_name = _sanitize_stage_name(stage_name)
        self.logger = logging.getLogger(PIPELINE_LOGGER_NAME)

    def __enter__(self):
        self.logger.info("START %s", self.stage_name)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            self.logger.info("SUCCESS %s", self.stage_name)
        else:
            self.logger.error("FAIL %s: %s", self.stage_name, exc_value)
        self.logger.info("END %s", self.stage_name)
        return False
