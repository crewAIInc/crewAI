"""Build the wheels and static files for running this checkout under Pyodide.

    uv run python lib/crewai-pyodide/build.py
    python -m http.server -d lib/crewai-pyodide/dist

Then open http://localhost:8000/ for the example app. ``dist/wheels`` can also
be passed to ``loadCrewAI({ wheelBaseURL })`` from any other page.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
LIB = HERE.parent
DIST = HERE / "dist"
WHEELS = DIST / "wheels"

# Built from this checkout; everything else comes from Pyodide's index or PyPI.
BOOTSTRAP_PACKAGE = "crewai-pyodide"
CREWAI_PACKAGES = ("crewai", "crewai-core")


def build_wheel(package: str) -> str:
    """Build ``lib/<package>`` into ``dist/wheels`` and return the wheel's name."""
    before = set(WHEELS.glob("*.whl"))
    subprocess.run(  # noqa: S603
        ["uv", "build", "--wheel", "--out-dir", str(WHEELS), str(LIB / package)],  # noqa: S607
        check=True,
    )
    (wheel,) = set(WHEELS.glob("*.whl")) - before
    return wheel.name


def main() -> None:
    shutil.rmtree(DIST, ignore_errors=True)
    WHEELS.mkdir(parents=True)

    manifest = {
        "bootstrap": build_wheel(BOOTSTRAP_PACKAGE),
        "crewai": [build_wheel(package) for package in CREWAI_PACKAGES],
    }
    (WHEELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.copy(HERE / "js" / "crewai-pyodide.mjs", DIST)
    shutil.copytree(HERE / "example", DIST, dirs_exist_ok=True)
    print(f"Built {DIST}")  # noqa: T201


if __name__ == "__main__":
    main()
