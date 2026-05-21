"""Remove intermediate columns."""

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pathlib import Path

import pandas as pd

from riskscape.config import paths


MEAN_COLUMNS = [
    # "sst_mean",
    # "chl_log_mean",
    # "ssh_mean",
    # "wind_speed_mean",
    # "chl",
    # "wind_u10",
    # "wind_v10",
    "adjusted_doy"
]


def main() -> int:
    root = paths["data"] / "features" / "environmental"

    for path in sorted(root.glob("year=*/part.parquet")):
        df = pd.read_parquet(path)

        drop_cols = [col for col in MEAN_COLUMNS if col in df.columns]

        if drop_cols:
            df = df.drop(columns=drop_cols)
            df.to_parquet(path, index=False, compression="zstd")
            print(f"Updated: {path}")
        else:
            print(f"No mean columns found: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())