"""Run correlation analysis."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.analysis.correlation import run_correlation_analysis


RUNS = {
    "env_features": {
        "source_tables": ["environmental"],
        "join_keys": ["h3", "date"],
        "features": [
            "sst",
            "chl",
            "chl_log",
            "ssh",
            "wind_speed",
        ],
        "method": "spearman",
        "plot": True,
    },
}


def main() -> int:
    for name, cfg in RUNS.items():
        run_correlation_analysis(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())