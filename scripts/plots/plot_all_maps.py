"""Run all map plotting scripts."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

MAP_SCRIPTS = [
    # "plot_study_area_map.py",
    # "plot_bathymetry_map.py",
    # "plot_species_presence_maps.py",
    # "plot_fishing_activity_map.py",
    "plot_plausibility_maps.py",
    "plot_prediction_maps.py",
]


def script_env() -> dict[str, str]:
    """Return environment with project src on PYTHONPATH."""
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{current}" if current else str(SRC_DIR)
    )

    return env


def run_script(script_name: str, env: dict[str, str]) -> None:
    """Run one plotting script."""
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Map script not found: {script_path}")

    print(f"\n=== Running {script_name} ===", flush=True)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> int:
    """Run all map plotting scripts."""
    env = script_env()

    for script_name in MAP_SCRIPTS:
        run_script(script_name, env)

    print("\nAll map scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
