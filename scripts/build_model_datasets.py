"""Build model-ready datasets."""

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
