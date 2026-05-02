"""Generate model predictions."""

from riskscape.model.predict import predict_models


def main() -> int:
    """Run prediction."""
    predict_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())