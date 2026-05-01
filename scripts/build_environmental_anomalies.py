"""Build environmental anomalies."""

from riskscape.features.anomalies import process_environmental_anomalies


def main() -> int:
    process_environmental_anomalies()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())