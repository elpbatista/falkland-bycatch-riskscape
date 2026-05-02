"""Build derived features."""

from riskscape.features.derived import process_environmental


def main() -> int:
    process_environmental()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())