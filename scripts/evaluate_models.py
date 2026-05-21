"""Evaluate trained models."""

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.model import evaluate_models


def main() -> int:
    """Run model evaluation."""
    setup_logging(stage="evaluate_models", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("evaluate_models"):
        evaluate_models()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
