"""Create model-ready tables from Layer 2."""

from pathlib import Path

import pandas as pd

from riskscape.config import cfg, paths


COLUMNS = [
    "date",
    "h3",
    "sst",
    "chl",
    "ssh",
    "sst_grad",
    "chl_grad",
    "ssh_grad",
    "sst_anom",
    "chl_anom",
    "ssh_anom",
    "wind",
    "wind_anom",
    "wind_grad",
]


def year_range():
    """Return inclusive year range from config."""

    start = pd.to_datetime(cfg["time"]["start"]).year
    end = pd.to_datetime(cfg["time"]["end"]).year
    return range(start, end + 1)


def main():
    """Write model-ready yearly tables."""

    src_dir = Path(paths["layer2"])
    dst_dir = Path(paths["data"]) / "model_ready"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for year in year_range():
        src = src_dir / f"year={year}.parquet"

        if not src.exists():
            raise FileNotFoundError(f"Missing Layer 2 file: {src}")

        print(f"Processing {year}")

        df = pd.read_parquet(src, columns=COLUMNS)

        dst = dst_dir / f"year={year}.parquet"
        df.to_parquet(dst, index=False)

        print(f"Saved: {dst}")
        print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()