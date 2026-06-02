"""Test that all public API classes are properly importable."""

import os
import subprocess
import sys


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput

    assert TaskOutput is not None


def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput

    assert CrewOutput is not None


def test_import_crewai_does_not_import_heavy_optional_deps():
    """`import crewai` must not eagerly import heavy optional dependencies.

    ``docling`` and its transitive dependencies (``torch``, ``transformers``) are
    optional and only needed when ``CrewDoclingSource`` is instantiated. Importing
    them at module load made ``import crewai`` slow and could fail outright on
    environments where the optional stack does not import cleanly (e.g. ``torch``
    on some CPython 3.13 builds). They must be imported lazily; this guards against
    a regression. A subprocess is used so the check is unaffected by modules other
    tests may have already imported into this process.
    """
    code = (
        "import sys\n"
        "import crewai  # noqa: F401\n"
        "heavy = [m for m in ('torch', 'docling', 'docling_core', 'transformers') if m in sys.modules]\n"
        "print(','.join(sorted(heavy)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true"},
    )
    leaked = result.stdout.strip()
    assert not leaked, (
        f"`import crewai` eagerly imported heavy optional dependencies: {leaked}. "
        "Import them lazily (only when actually used)."
    )
