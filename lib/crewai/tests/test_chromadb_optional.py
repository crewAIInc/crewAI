"""ChromaDB stays off the default import path."""

import subprocess
import sys


def test_importing_crewai_does_not_import_chromadb() -> None:
    """A plain ``import crewai`` succeeds when chromadb is not installed."""
    script = """
import sys

sys.modules["chromadb"] = None
import crewai

assert "chromadb" not in sys.modules or sys.modules["chromadb"] is None
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
