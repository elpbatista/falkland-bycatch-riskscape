"""Build species presence feature tables."""

from riskscape.features import build_species_presence_features
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run species presence feature generation."""
    setup_logging(stage="build_species_presence_features", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_species_presence_features"):
        build_species_presence_features()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())