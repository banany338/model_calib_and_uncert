"""Convenience wrapper for running the project without installing the package."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disaster_uncertainty.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
