"""Fix environmental date dtype to UTC."""

import pandas as pd

from riskscape.config import paths


def main() -> int:
    # root = paths["data"] / "features" / "environmental"
    root = paths["data"] / "features" / "fishing_effort"

    for path in sorted(root.glob("year=*/part.parquet")):
        df = pd.read_parquet(path)

        df["date"] = pd.to_datetime(df["date"], utc=True)

        df.to_parquet(path, index=False, compression="zstd")

        print(f"Updated: {path}")
        print(df["date"].dtype)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())