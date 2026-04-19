"""Build the H3 grid defined in the active configuration."""

import logging

from riskscape.grid import build_h3_grid
from riskscape.logs import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    """Run grid generation."""
    setup_logging(stage="build_grid", verbose=True)
    logger.info("Starting grid generation")

    build_h3_grid()
    logger.info("Grid generation complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())