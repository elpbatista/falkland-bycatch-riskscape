"""Train models."""

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.model.train import train_models


def main() -> int:
    """Run model training."""
    setup_logging(stage="train_models", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("train_models"):
        train_models()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
