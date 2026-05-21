"""Shared layout helpers for 12-panel monthly map figures."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from riskscape.visualization.base_map import MapBounds, OCEAN_COLOR


@dataclass(frozen=True)
class MonthlyMapLayout:
    """Layout settings for the standard monthly map matrix."""

    nrows: int = 4
    ncols: int = 3
    figsize: tuple[float, float] = (10.5, 16.0)
    title_fontsize: int = 16
    panel_title_fontsize: int = 11
    panel_margin: float = 0.35
    suptitle_y: float = 0.985
    left: float = 0.025
    right: float = 0.84
    top: float = 0.95
    bottom: float = 0.035
    wspace: float = 0.15
    hspace: float = 0.15
    colorbar_left: float = 0.88
    colorbar_bottom: float = 0.20
    colorbar_width: float = 0.025
    colorbar_height: float = 0.60
    dpi: int = 300


def month_panel_title(month: int) -> str:
    """Return the standard title for one monthly panel."""
    return calendar.month_abbr[month]


def create_monthly_map_grid(
    title: str,
    layout: MonthlyMapLayout | None = None,
) -> tuple[plt.Figure, list[Axes]]:
    """Create the standard 12-panel monthly map grid."""
    layout = layout or MonthlyMapLayout()
    fig, axes = plt.subplots(
        nrows=layout.nrows,
        ncols=layout.ncols,
        figsize=layout.figsize,
        constrained_layout=False,
    )
    axes_flat = cast(list[Axes], axes.ravel().tolist())
    fig.suptitle(title, fontsize=layout.title_fontsize, y=layout.suptitle_y)
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    return fig, axes_flat


def format_month_panel(
    ax: Axes,
    month: int,
    bounds: MapBounds | None = None,
    layout: MonthlyMapLayout | None = None,
) -> None:
    """Apply standard map panel extent, face color, title, and ticks."""
    layout = layout or MonthlyMapLayout()
    bounds = bounds or MapBounds.from_config()
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=layout.panel_margin)
    ax.set_title(month_panel_title(month), fontsize=layout.panel_title_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def month_axes(axes: Iterable[Axes]) -> Iterable[tuple[int, Axes]]:
    """Yield one-based month numbers with their corresponding axes."""
    return enumerate(axes, start=1)


def add_monthly_colorbar_axis(
    fig: plt.Figure,
    layout: MonthlyMapLayout | None = None,
) -> Axes:
    """Add the standard right-side colorbar axis."""
    layout = layout or MonthlyMapLayout()
    return fig.add_axes(
        (
            layout.colorbar_left,
            layout.colorbar_bottom,
            layout.colorbar_width,
            layout.colorbar_height,
        )
    )


def add_centered_colorbar_axis(
    fig: plt.Figure,
    segment_count: int,
    layout: MonthlyMapLayout | None = None,
) -> Axes:
    """Add a compact centered colorbar axis for discrete legends."""
    layout = layout or MonthlyMapLayout()
    fig_width, fig_height = fig.get_size_inches()
    segment_height = layout.colorbar_width * fig_width / fig_height
    colorbar_height = segment_height * segment_count
    colorbar_bottom = 0.50 - colorbar_height / 2
    return fig.add_axes(
        (
            layout.colorbar_left,
            colorbar_bottom,
            layout.colorbar_width,
            colorbar_height,
        )
    )


def save_monthly_map(
    fig: plt.Figure,
    out_file: Path,
    layout: MonthlyMapLayout | None = None,
) -> Path:
    """Save and close a monthly map figure."""
    layout = layout or MonthlyMapLayout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=layout.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file
