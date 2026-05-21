"""Summarize yearly feature table quality."""

from riskscape.logs import setup_logging, setup_pipeline_logging, stage_context
from riskscape.qa import run_feature_qa_summary


def main() -> int:
    """Run feature QA summary."""
    setup_logging(stage="feature_qa_summary", verbose=True)
    setup_pipeline_logging(verbose=True)

    with stage_context("feature_qa_summary"):
        run_feature_qa_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
