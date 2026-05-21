"""Build environmental anomalies."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.features.anomalies import process_environmental_anomalies


def main() -> int:
    process_environmental_anomalies()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())