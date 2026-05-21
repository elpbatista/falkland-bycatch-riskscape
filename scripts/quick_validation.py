"""Run lightweight validation checks over generated products."""

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
