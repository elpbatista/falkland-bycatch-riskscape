"""
Project configuration loader.

Loads config.yaml (or overridden config) and resolves all paths
relative to the project root directory.
"""

from pathlib import Path
import os
import yaml


def load_dotenv(env_file: Path) -> None:
    """
    Load simple KEY=VALUE pairs from a local .env file.
    """
    if not env_file.exists():
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if key and key not in os.environ:
                os.environ[key] = value


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
load_dotenv(PROJECT_ROOT / ".env")

# Select config file (override or default)
CONFIG_NAME = os.environ.get("RISKCAPE_CONFIG", "config.yaml")
CONFIG_FILE = PROJECT_ROOT / CONFIG_NAME

if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"{CONFIG_NAME} not found in project root")

# Load configuration
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
