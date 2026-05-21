"""Generate model predictions."""

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
