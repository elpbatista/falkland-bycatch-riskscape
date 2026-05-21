"""Summarize yearly feature table quality."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.qa import run_feature_qa_summary


def main() -> int:
    """Run feature QA summary."""
    setup_logging(stage="feature_qa_summary", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("feature_qa_summary"):
        run_feature_qa_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
