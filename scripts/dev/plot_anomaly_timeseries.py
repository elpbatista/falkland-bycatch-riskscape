"""
Plot basin-wide mean anomaly time series (2014–2023).
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from riskscape.config import cfg, paths


def load_all_years():

    start_year = pd.to_datetime(cfg["time"]["start"]).year
    end_year = pd.to_datetime(cfg["time"]["end"]).year

    frames = []

    for year in range(start_year, end_year + 1):
        file_path = Path(paths["data"]) / "layer2" / f"year={year}.parquet"
        df = pd.read_parquet(
            file_path,
            columns=["date", "sst_anom", "chl_anom", "ssh_anom"]
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():

    print("Loading Layer 2 anomalies")
    df = load_all_years()

    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby("date", as_index=False)
        .mean()
        .sort_values("date")
    )

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(daily["date"], daily["sst_anom"])
    ax[0].set_title("Mean SST Anomaly")

    ax[1].plot(daily["date"], daily["chl_anom"])
    ax[1].set_title("Mean Chlorophyll Anomaly")

    ax[2].plot(daily["date"], daily["ssh_anom"])
    ax[2].set_title("Mean SSH Anomaly")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()