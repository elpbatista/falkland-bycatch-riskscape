"""Build the H3 grid defined in the active configuration."""

import logging

from riskscape.grid import build_h3_grid, convert_grid_to_uint64
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


logger = logging.getLogger(__name__)


def main() -> int:
    """Run grid generation and uint64 conversion."""
    setup_logging(stage="build_grid", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_grid"):
        build_h3_grid()

    with stage_context("convert_grid_to_uint64"):
        convert_grid_to_uint64()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())