"""
Project configuration loader.

Loads config.yaml and resolves all paths relative to the
project root directory.
"""

from pathlib import Path
import yaml


def find_project_root() -> Path:
    """
    Find the project root by locating config.yaml.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return parent

    raise FileNotFoundError("config.yaml not found in project tree")


# Resolve project root
PROJECT_ROOT = find_project_root()

# Load configuration
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path from config relative to project root.
    """
    return (PROJECT_ROOT / relative_path).resolve()


# Resolve configured paths
paths = {
    key: resolve_path(value)
    for key, value in cfg.get("paths", {}).items()
}