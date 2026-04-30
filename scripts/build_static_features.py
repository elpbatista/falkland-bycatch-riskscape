"""Build static feature table."""

from riskscape.features import build_static_features
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run static feature generation."""
    setup_logging(stage="build_static_features", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_static_features"):
        build_static_features()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())