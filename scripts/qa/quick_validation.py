"""Run lightweight validation checks over generated products."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.qa import run_quick_validation


def main() -> int:
    """Run quick validation checks."""
    setup_logging(stage="quick_validation", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("quick_validation"):
        try:
            run_quick_validation()
        except RuntimeError as exc:
            print(exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
