"""Run high-level riskscape workflow stages."""

from __future__ import annotations

import argparse

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.workflow import STAGES, run_workflow


PUBLIC_STAGES = [
    *STAGES.keys(),
    "all",
    "all-with-downloads",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run high-level riskscape workflow stages",
    )
    parser.add_argument(
        "--stage",
        choices=PUBLIC_STAGES,
        default="all",
        help=(
            "Workflow stage to run. 'all' starts after data have been restored "
            "and downloaded; 'all-with-downloads' includes external downloads."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    """Run selected workflow stage."""
    args = parse_args()

    setup_logging(stage="run_pipeline", verbose=args.verbose)
    setup_pipeline_logging(verbose=args.verbose)

    with stage_context("run_pipeline"):
        run_workflow(args.stage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
