"""Repository root. Modules resolve paths against it; the repo is used as a source checkout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
