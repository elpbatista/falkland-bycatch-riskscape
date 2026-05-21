"""Shared legend and colorbar behavior for map figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from matplotlib import colors
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np


LegendMode = Literal[
    "continuous",
    "continuous_inverted",
    "diverging_centered",
    "binned_quantile",
    "categorical",
]


@dataclass(frozen=True)
class LegendStyle:
    """Colorbar/legend behavior independent from map layout."""

    mode: LegendMode = "continuous"
    title: str | None = None
    bottom_label: str = "Low"
    top_label: str = "High"
    labels: tuple[str, ...] | None = None
    invert: bool = False
    draw_edges: bool = False
    shrink: float = 0.72
    pad: float = 0.02
    fraction: float = 0.035

    @property
    def is_binned(self) -> bool:
        """Return whether this style requires discrete color bins."""
        return self.mode in {"binned_quantile", "categorical"}

    @property
    def should_invert(self) -> bool:
        """Return whether the colorbar direction should be inverted."""
        return self.invert or self.mode == "continuous_inverted"


def label_colorbar_extremes(
    cbar,
    bottom: str = "Low",
    top: str = "High",
) -> None:
    """Label a colorbar with simple endpoint text."""
    cbar.ax.text(
        0.5,
        -0.03,
        bottom,
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    cbar.ax.text(
        0.5,
        1.03,
        top,
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_continuous_colorbar(
    ax,
    cmap: str | colors.Colormap,
    norm: colors.Normalize,
    legend: LegendStyle,
    cax=None,
) -> None:
    """Draw a continuous low/high colorbar."""
    kwargs = {
        "label": legend.title,
        "ticks": [],
    }
    if cax is None:
        kwargs.update(
            {
                "ax": ax,
                "shrink": legend.shrink,
                "pad": legend.pad,
                "fraction": legend.fraction,
            }
        )
    else:
        kwargs["cax"] = cax

    colormap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=colormap), **kwargs)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(which="both", length=0)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("face")

    if legend.should_invert:
        cbar.ax.invert_yaxis()

    label_colorbar_extremes(
        cbar,
        bottom=legend.bottom_label,
        top=legend.top_label,
    )


def draw_binned_colorbar(
    ax,
    cmap: colors.Colormap,
    norm: colors.BoundaryNorm,
    legend: LegendStyle,
    color_scale: str = "linear",
    cax=None,
) -> None:
    """Draw a discrete labeled colorbar."""
    if legend.labels is None:
        raise ValueError("Binned colorbar requires legend labels")

    boundaries = norm.boundaries
    ticks = (boundaries[:-1] + boundaries[1:]) / 2

    if color_scale == "log":
        ticks = np.sqrt(boundaries[:-1] * boundaries[1:])

    kwargs = {
        "label": legend.title,
        "ticks": ticks,
        "spacing": "uniform",
        "drawedges": legend.draw_edges,
    }
    if cax is None:
        kwargs.update(
            {
                "ax": ax,
                "shrink": 0.28,
                "aspect": len(legend.labels),
                "pad": legend.pad,
                "fraction": legend.fraction,
            }
        )
    else:
        kwargs["cax"] = cax

    cbar = ax.figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        **kwargs,
    )
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("#8a8a8a")
    cbar.outline.set_linewidth(0.8)
    if legend.draw_edges:
        cbar.dividers.set_color("#8a8a8a")
        cbar.dividers.set_linewidth(0.6)
    cbar.ax.set_yticklabels(legend.labels)
    cbar.ax.tick_params(which="both", length=0)
    cbar.ax.minorticks_off()
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("face")


def draw_map_colorbar(
    ax,
    cmap: str | colors.Colormap,
    norm: colors.Normalize,
    legend: LegendStyle,
    color_scale: str = "linear",
    cax=None,
) -> None:
    """Draw a map colorbar using an explicit legend mode."""
    if legend.is_binned:
        if not isinstance(norm, colors.BoundaryNorm):
            raise TypeError("Binned legends require BoundaryNorm")
        draw_binned_colorbar(
            ax=ax,
            cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap,
            norm=norm,
            legend=legend,
            color_scale=color_scale,
            cax=cax,
        )
        return

    draw_continuous_colorbar(
        ax=ax,
        cmap=cmap,
        norm=norm,
        legend=legend,
        cax=cax,
    )
