"""Build the seasonal lookup table."""

from riskscape.indices import build_seasonal_lookup
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run seasonal lookup generation."""
    setup_logging(stage="build_seasonal_lookup", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_seasonal_lookup"):
        build_seasonal_lookup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())