"""Build H3 lookup tables."""

from riskscape.indices import build_h3_lookup
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run lookup generation."""
    setup_logging(stage="build_h3_lookup", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_h3_lookup"):
        build_h3_lookup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())