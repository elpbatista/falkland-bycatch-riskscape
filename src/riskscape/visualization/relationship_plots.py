"""Relationship diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_binned_relationships(
    binned: pd.DataFrame,
    out_dir: Path,
    groups: list[str],
) -> None:
    """Plot binned diagnostic relationships."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if binned.empty:
        return

    group_cols = groups if groups else [None]

    for predictor in sorted(binned["predictor"].unique()):
        data = binned[binned["predictor"] == predictor].copy()

        if groups:
            grouped = data.groupby(groups)
        else:
            grouped = [("all", data)]

        for group_name, group_df in grouped:
            if isinstance(group_name, tuple):
                label = "_".join(str(v) for v in group_name)
            else:
                label = str(group_name)

            fig, ax = plt.subplots(figsize=(8, 5))

            # ax.plot(
            #     group_df["predictor_mean"],
            #     group_df["response_mean"],
            #     marker="o",
            # )

            sizes = group_df["response_count"]

            ax.scatter(
                group_df["predictor_mean"],
                group_df["response_mean"],
                s=10 + 2 * sizes,
)

            ax.set_xlabel(predictor)
            ax.set_ylabel("Mean response")
            ax.set_title(f"{label}: response vs {predictor}")

            out_file = out_dir / f"binned_{label}_{predictor}.png"
            fig.savefig(out_file, dpi=200, bbox_inches="tight")
            plt.close(fig)