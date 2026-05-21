"""Download public reference layers used by the riskscape workflow."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse

from riskscape.downloads.reference_data import DATASETS, download_reference_datasets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download public reference layers for the riskscape workflow",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASETS),
        help="Reference dataset(s) to download. Defaults to all datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reference folders before extracting downloads.",
    )
    return parser.parse_args()


def main() -> int:
    """Download selected reference datasets."""
    args = parse_args()
    return download_reference_datasets(
        dataset_names=args.dataset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
