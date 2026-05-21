"""Run relationships analysis."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.analysis.relationships import run_relationship_analysis


RUNS = {
    "species_env_BBAL": {
        "response_table": "species_presence",
        "response_expr": "presence_count / individual_count",
        "response_name": "residence_index",
        "join_keys": ["h3", "date"],
        "groups": ["species"],
        "response_filter": {"species": "BBAL"},
        "predictor_tables": ["environmental"],
        "predictors": [
            "sst",
            "chl",
            "chl_log",
            "ssh",
            "wind_speed",
        ],
        "bins": 20,
    },
}


def main() -> int:
    for name, cfg in RUNS.items():
        run_relationship_analysis(name, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())