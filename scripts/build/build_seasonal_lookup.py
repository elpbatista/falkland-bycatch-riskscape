"""Build the seasonal lookup table."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.indices import build_seasonal_lookup
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run seasonal lookup generation."""
    setup_logging(stage="build_seasonal_lookup", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_seasonal_lookup"):
        build_seasonal_lookup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())