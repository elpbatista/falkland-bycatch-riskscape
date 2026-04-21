"""Convert H3 grid ids to uint64."""

import logging

from riskscape.grid import convert_grid_to_uint64
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context

logger = logging.getLogger(__name__)

def main() -> int:
    """Run grid conversion."""
    setup_logging(stage="convert_grid_to_uint64", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("convert_grid_to_uint64"):
        convert_grid_to_uint64()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
  