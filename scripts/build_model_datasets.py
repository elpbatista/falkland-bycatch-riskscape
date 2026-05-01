"""Build model-ready datasets."""

from riskscape.model import build_model_datasets


def main() -> int:
    """Run model dataset generation."""
    build_model_datasets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())