"""Build environmental gradients."""

from riskscape.features.gradients import process_environmental_gradients


def main() -> int:
    process_environmental_gradients()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())