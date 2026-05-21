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
import sys

from riskscape.downloads.pipeline import download_configured_datasets
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


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
        return download_configured_datasets(
            dataset_names=args.dataset,
            clean=args.clean,
        )


if __name__ == "__main__":
    sys.exit(main())
