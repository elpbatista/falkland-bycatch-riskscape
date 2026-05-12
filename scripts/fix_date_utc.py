"""Fix date dtype to timezone-free UTC calendar days."""

import pandas as pd

from riskscape.config import paths
from riskscape.utils.dates import normalize_date_column


def main() -> int:
    # root = paths["data"] / "features" / "environmental"
    root = paths["data"] / "features" / "fishing_effort"

    for path in sorted(root.glob("year=*/part.parquet")):
        df = pd.read_parquet(path)

        df = normalize_date_column(df)

        df.to_parquet(path, index=False, compression="zstd")

        print(f"Updated: {path}")
        print(df["date"].dtype)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
