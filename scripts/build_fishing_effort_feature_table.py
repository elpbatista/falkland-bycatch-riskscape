"""Build fishing effort feature tables."""

from riskscape.features import build_fishing_effort_features
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run fishing effort feature generation."""
    setup_logging(stage="build_fishing_effort_features", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_fishing_effort_features"):
        build_fishing_effort_features()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())