"""Train the active production species-use model."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.model.train import train_production_extra_trees_model


def main() -> int:
    """Run active production species model training."""
    train_production_extra_trees_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
