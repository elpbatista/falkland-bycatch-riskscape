"""Build derived features."""

from riskscape.features.derived import process_environmental
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run derived feature generation."""
    setup_logging(stage="build_derived_features", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_derived_features"):
        process_environmental()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
