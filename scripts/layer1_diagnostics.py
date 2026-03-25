"""Layer 1 diagnostics: daily grid coverage check."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from riskscape.config import cfg, paths


def load_grid():
    """Load H3 grid geometry."""

    resolution = cfg["grid"]["resolution"]
    region = cfg["region"]["name"]

    grid_file = paths["grids"] / f"h3_res{resolution}_{region}.geojson"

    grid = gpd.read_file(grid_file)
    grid["h3"] = grid["id"].apply(lambda x: int(x, 16)).astype("uint64")

    return grid


def load_layer1_year(year):
    """Load one Layer 1 yearly parquet."""

    path = Path(paths["layer1"]) / f"year={year}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=["date", "h3", "sst", "chl", "ssh"])
    df["date"] = pd.to_datetime(df["date"])

    return df


def daily_coverage_summary(df, grid, day_str):
    """Print daily row and grid coverage summary."""

    day = pd.to_datetime(day_str).date()
    day_df = df[df["date"].dt.date == day].copy()

    total_grid = len(grid)
    rows_day = len(day_df)
    unique_h3_day = day_df["h3"].nunique()

    missing_h3 = set(grid["h3"]) - set(day_df["h3"])
    extra_h3 = set(day_df["h3"]) - set(grid["h3"])

    print(f"\n=== Daily coverage summary for {day} ===")
    print(f"Rows that day: {rows_day}")
    print(f"Unique H3 cells that day: {unique_h3_day}")
    print(f"Total grid cells: {total_grid}")
    print(f"Missing grid cells that day: {len(missing_h3)}")
    print(f"Extra cells not in grid: {len(extra_h3)}")

    return day, day_df, missing_h3


def plot_missing_cells(grid, day_df, falklands, day, plots_dir):
    """Plot daily coverage and missing cells."""

    grid_day = grid.merge(
        day_df[["h3"]].assign(has_data=1),
        on="h3",
        how="left",
    )

    grid_day["has_data"] = grid_day["has_data"].fillna(0).astype("int8")

    fig, ax = plt.subplots(figsize=(12, 10))

    grid_day.plot(
        column="has_data",
        cmap="viridis",
        linewidth=0,
        legend=True,
        ax=ax,
        vmin=0,
        vmax=1,
    )

    if falklands is not None:
        falklands.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
        )

    ax.set_title(
        f"Layer 1 daily coverage on {day}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()

    out_file = plots_dir / f"layer1_coverage_{day}.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_file}")


def plot_variable_nan_for_day(grid, day_df, falklands, day, plots_dir):
    """Plot per-variable NaN pattern for a specific day."""

    for col in ["sst", "chl", "ssh"]:
        temp = day_df[["h3", col]].copy()
        temp[f"{col}_nan"] = temp[col].isna().astype("int8")
        temp = temp[["h3", f"{col}_nan"]]

        grid_col = grid.merge(temp, on="h3", how="left")
        grid_col[f"{col}_nan"] = grid_col[f"{col}_nan"].fillna(1).astype("int8")

        fig, ax = plt.subplots(figsize=(12, 10))

        grid_col.plot(
            column=f"{col}_nan",
            cmap="viridis",
            linewidth=0,
            legend=True,
            ax=ax,
            vmin=0,
            vmax=1,
        )

        if falklands is not None:
            falklands.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
            )

        ax.set_title(
            f"{col} NaN pattern on {day}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()

        out_file = plots_dir / f"{col}_nan_{day}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_file}")


def load_falklands():
    """Load Falkland Islands boundary if available."""

    falklands_file = paths.get("falklands")
    if not falklands_file:
        print("Warning: Falklands boundary not configured")
        return None

    gdf = gpd.read_file(falklands_file)

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def main():
    year = 2022
    day_str = "2022-07-02"

    plots_dir = paths.get("plots", paths["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading grid")
    grid = load_grid()

    print("Loading Layer 1")
    df = load_layer1_year(year)

    print("Loading Falkland Islands boundary")
    falklands = load_falklands()

    day, day_df, _ = daily_coverage_summary(df, grid, day_str)

    plot_missing_cells(grid, day_df, falklands, day, plots_dir)
    plot_variable_nan_for_day(grid, day_df, falklands, day, plots_dir)

    print("\nDone")


if __name__ == "__main__":
    main()