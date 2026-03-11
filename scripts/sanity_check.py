# import pandas as pd

# df = pd.read_parquet("data/layer2/year=2014.parquet")

# print(df.columns)
# print(df[["sst_grad", "chl_grad", "ssh_grad"]].describe())

"""
Sanity check Layer 2 anomalies.
"""

from pathlib import Path

import pandas as pd

from riskscape.config import cfg, paths


def main():

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    for year in range(start_year, end_year + 1):

        print(f"\nChecking {year}")

        file_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"

        df = pd.read_parquet(file_path)

        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())

        print("\nAnomaly statistics:")
        print(
            df[["sst_anom", "chl_anom", "ssh_anom"]]
            .describe()
        )

        print("\nMean anomalies:")
        print(
            df[["sst_anom", "chl_anom", "ssh_anom"]]
            .mean()
        )


if __name__ == "__main__":
    main()