"""Utilities for managing project log files."""

from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_logs_dir(logs_dir: Optional[Path] = None) -> Path:
    """Return the logs directory, creating it if needed."""
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parents[3] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def rotate_log_on_date_change(log_file: Path) -> None:
    """Archive log file if it's from a previous day, keeping only one backup."""
    if not log_file.exists():
        return

    # Get the modification date of the existing log file
    log_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
    log_date = log_mtime.date()
    today = datetime.now().date()

    # If the log is from today, no rotation needed
    if log_date == today:
        return

    # Archive the old log with a date suffix
    archived_name = f"{log_file.stem}-{log_date.isoformat()}{log_file.suffix}"
    archived_path = log_file.with_name(archived_name)
    log_file.replace(archived_path)

    # Clean up any other archived logs, keeping only the most recent
    logs_dir = log_file.parent
    pattern = f"{log_file.stem}-*.log"
    archives = sorted(logs_dir.glob(pattern))

    if len(archives) > 1:
        for old_archive in archives[:-1]:  # Keep only the latest archive
            old_archive.unlink()


