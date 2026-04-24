"""Build environmental feature tables."""

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