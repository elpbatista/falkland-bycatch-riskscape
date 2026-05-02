"""Train models."""

from riskscape.model.train import train_models


def main() -> int:
    train_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())