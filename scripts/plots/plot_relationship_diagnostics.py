"""Plot relationship diagnostic outputs."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from riskscape.config import paths
from riskscape.visualization.relationship_plots import plot_binned_relationships


PLOT_RUNS = {
    # "species_static_relationships": {
    #     "groups": ["species"],
    # },
    "species_environmental_relationships": {
        "groups": ["species"],
    },
    # "fishing_static_relationships": {
    #     "groups": [],
    # },
}


def main() -> int:
    """Plot configured relationship diagnostics."""
    root = paths["data"] / "diagnostics"

    for run_name, run_cfg in PLOT_RUNS.items():
        run_root = root / run_name
        table_path = run_root / "tables" / "binned_summary.parquet"

        if not table_path.exists():
            print("Skipping missing diagnostics:", run_name)
            continue

        binned = pd.read_parquet(table_path)
        out_dir = run_root / "figures"

        plot_binned_relationships(
            binned=binned,
            out_dir=out_dir,
            groups=run_cfg.get("groups", []),
        )

        print("Saved figures:", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())