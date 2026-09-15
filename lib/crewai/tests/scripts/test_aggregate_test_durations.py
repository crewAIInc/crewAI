from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[4] / "scripts" / "aggregate_test_durations.py"
SPEC = importlib.util.spec_from_file_location("aggregate_test_durations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_aggregates_repeated_profiles_by_suite_and_nodeid(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "crewai-durations.json"
    second = tmp_path / "second" / "crewai-durations.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(
        json.dumps(
            {
                "tests/test_fast.py::test_fast": 0.1,
                "tests/test_slow.py::test_slow": 4.0,
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "tests/test_fast.py::test_fast": 0.3,
                "tests/test_slow.py::test_slow": 6.0,
            }
        )
    )

    rows = MODULE.summarize([tmp_path])

    assert rows == [
        {
            "suite": "crewai",
            "nodeid": "tests/test_slow.py::test_slow",
            "samples": 2,
            "mean_seconds": 5.0,
            "median_seconds": 5.0,
            "p95_seconds": 6.0,
            "max_seconds": 6.0,
        },
        {
            "suite": "crewai",
            "nodeid": "tests/test_fast.py::test_fast",
            "samples": 2,
            "mean_seconds": 0.2,
            "median_seconds": 0.2,
            "p95_seconds": 0.3,
            "max_seconds": 0.3,
        },
    ]
