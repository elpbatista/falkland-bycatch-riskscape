"""Generate model predictions."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.model.predict import predict_models


def main() -> int:
    """Run prediction."""
    setup_logging(stage="predict_models", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("predict_models"):
        predict_models()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
