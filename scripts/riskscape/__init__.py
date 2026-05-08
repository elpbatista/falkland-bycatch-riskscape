"""Import shim for running project scripts directly.

When a script is executed as ``python3 scripts/<name>.py``, Python adds the
``scripts`` directory to ``sys.path`` but not the project ``src`` directory.
This package points ``import riskscape`` at the real package under ``src``.
"""

from __future__ import annotations

from pathlib import Path


__path__ = [
    str(Path(__file__).resolve().parents[2] / "src" / "riskscape")
]
