"""Train the active production species-use model."""

from riskscape.model.train import train_production_extra_trees_model


def main() -> int:
    """Run active production species model training."""
    train_production_extra_trees_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
