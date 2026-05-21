"""Inspect a generated parquet table."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse

from riskscape.qa import inspect_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect a generated table")
    parser.add_argument(
        "path",
        nargs="?",
        default="static",
        help=(
            "Parquet path or alias. Aliases include environmental, static, "
            "fishing_training, species_training, and predictions."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="Number of head rows to print.",
    )
    return parser.parse_args()


def main() -> int:
    """Inspect selected table."""
    args = parse_args()
    inspect_table(args.path, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
