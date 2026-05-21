"""Build environmental feature tables."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.features import build_environmental_features
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run environmental feature generation."""
    setup_logging(stage="build_environmental_features", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_environmental_features"):
        build_environmental_features()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())