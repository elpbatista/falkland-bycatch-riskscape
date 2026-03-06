"""Plot 10-year SST time series for the Falkland Islands grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import paths


def main():

    layer1_dir = paths.get("layer1", paths["data"] / "layer1")

    print("Loading yearly files")

    files = sorted(layer1_dir.glob("year=*.parquet"))

    dfs = []

    for f in files:
        print("Reading", f.name)
        dfs.append(pd.read_parquet(f, columns=["date", "sst"]))

    df = pd.concat(dfs, ignore_index=True)

    # Convert to Celsius
    df["sst"] = df["sst"] - 273.15

    print("Computing daily mean over Falkland grid")

    ts = (
        df.groupby("date")["sst"]
        .mean()
        .sort_index()
    )

    print("Plotting")

    fig, ax = plt.subplots(figsize=(12, 5))

    ts.plot(ax=ax, linewidth=1)

    ax.set_title("Sea Surface Temperature — Falkland Islands (2014–2023)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Date")

    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()