"""Build indexed H3 neighbor table."""

from riskscape.indices import build_neighbor_index_table
from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context


def main() -> int:
    """Run indexed neighbor table generation."""
    setup_logging(stage="build_neighbor_index_table", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("build_neighbor_index_table"):
        build_neighbor_index_table()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())