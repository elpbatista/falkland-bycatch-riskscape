"""Shared weekly map layout and animation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Iterable, cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from riskscape.visualization.base_map import MapBounds, OCEAN_COLOR


@dataclass(frozen=True)
class WeeklyMapLayout:
    """Layout settings for weekly small-multiple maps."""

    panel_width: float = 4.0
    panel_height: float = 5.1
    frame_size: tuple[float, float] = (7.2, 8.4)
    panel_margin: float = 0.35
    title_fontsize: int = 16
    panel_title_fontsize: int = 10
    frame_title_fontsize: int = 13
    suptitle_y: float = 0.985
    left: float = 0.02
    right: float = 0.91
    top: float = 0.93
    bottom: float = 0.035
    wspace: float = 0.09
    hspace: float = 0.16
    colorbar_position: tuple[float, float, float, float] = (
        0.935,
        0.43,
        0.018,
        0.14,
    )
    frame_left: float = 0.02
    frame_right: float = 0.88
    frame_top: float = 0.94
    frame_bottom: float = 0.02
    dpi: int = 300


def week_panel_title(species: str, week: int) -> str:
    """Return the standard title for a weekly panel."""
    return f"{species} — ISO week {week:02d}"


def format_week_panel(
    ax: Axes,
    species: str,
    week: int,
    bounds: MapBounds | None = None,
    layout: WeeklyMapLayout | None = None,
    title: str | None = None,
) -> None:
    """Apply standard weekly panel extent, face color, title, and ticks."""
    layout = layout or WeeklyMapLayout()
    bounds = bounds or MapBounds.from_config()
    ax.set_facecolor(OCEAN_COLOR)
    bounds.apply_to_axis(ax, margin=layout.panel_margin)
    ax.set_title(
        title or week_panel_title(species, week),
        fontsize=layout.panel_title_fontsize,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def create_weekly_map_grid(
    species_values: list[str],
    weeks: list[int],
    title: str,
    layout: WeeklyMapLayout | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create the standard species-by-week small-multiple grid."""
    layout = layout or WeeklyMapLayout()
    fig, axes = plt.subplots(
        nrows=len(species_values),
        ncols=len(weeks),
        figsize=(
            layout.panel_width * len(weeks),
            layout.panel_height * len(species_values),
        ),
        constrained_layout=False,
    )
    axes_array = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=layout.title_fontsize, y=layout.suptitle_y)
    fig.subplots_adjust(
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    return fig, axes_array


def weekly_axes(
    axes_array: np.ndarray,
    species_values: list[str],
    weeks: list[int],
) -> Iterable[tuple[Axes, str, int]]:
    """Yield axes with species and week values."""
    for row, species in enumerate(species_values):
        for col, week in enumerate(weeks):
            yield cast(Axes, axes_array[row, col]), species, week


def add_weekly_colorbar_axis(
    fig: plt.Figure,
    layout: WeeklyMapLayout | None = None,
) -> Axes:
    """Add the standard compact weekly colorbar axis."""
    layout = layout or WeeklyMapLayout()
    return fig.add_axes(layout.colorbar_position)


def create_weekly_frame(
    layout: WeeklyMapLayout | None = None,
) -> tuple[plt.Figure, Axes]:
    """Create one animation frame figure."""
    layout = layout or WeeklyMapLayout()
    return plt.subplots(figsize=layout.frame_size, constrained_layout=False)


def format_weekly_frame(
    fig: plt.Figure,
    layout: WeeklyMapLayout | None = None,
) -> None:
    """Apply the standard frame margins."""
    layout = layout or WeeklyMapLayout()
    fig.subplots_adjust(
        left=layout.frame_left,
        right=layout.frame_right,
        top=layout.frame_top,
        bottom=layout.frame_bottom,
    )


def save_weekly_map(
    fig: plt.Figure,
    out_file: Path,
    layout: WeeklyMapLayout | None = None,
    dpi: int | None = None,
) -> Path:
    """Save and close a weekly map figure."""
    layout = layout or WeeklyMapLayout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi or layout.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def encode_mp4(frame_dir: Path, out_file: Path, fps: int) -> None:
    """Encode numbered weekly PNG frames into MP4 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(candidate).exists():
                ffmpeg = candidate
                break

    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode MP4 animations")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frame_dir / "week_*.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
