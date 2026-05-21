"""Plot study area with fisheries and conservation-zone reference layers."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from riskscape.visualization.study_area import plot_study_area_map


def main() -> int:
    """Run study area map plot."""
    out_file = plot_study_area_map()
    print(f"Saved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
