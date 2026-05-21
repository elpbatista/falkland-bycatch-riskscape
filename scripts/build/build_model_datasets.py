"""Build model-ready datasets."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.model import build_model_datasets


def main() -> int:
    """Run model dataset generation."""
    setup_logging(stage="build_model_datasets", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_model_datasets"):
        build_model_datasets()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
