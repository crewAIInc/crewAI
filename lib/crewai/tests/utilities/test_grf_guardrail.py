from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "crewai"
    / "utilities"
    / "grf_guardrail.py"
)
_SPEC = importlib.util.spec_from_file_location("grf_guardrail_under_test", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load GRF guardrail module at {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class GrfGuardrailTests(unittest.TestCase):
    def test_failed_local_check_returns_actionable_retry_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "grf.py").write_text(
                "import sys\n"
                "print('acceptance is incomplete', file=sys.stderr)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )

            passed, feedback = _MODULE.grf_guardrail(project)(None)

        self.assertFalse(passed)
        self.assertIn("acceptance is incomplete", feedback)

    def test_corrected_local_check_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            acceptance = project / "acceptance.txt"
            acceptance.write_text("missing", encoding="utf-8")
            (project / "grf.py").write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "if Path('acceptance.txt').read_text() != 'complete':\n"
                "    print('acceptance evidence is incomplete', file=sys.stderr)\n"
                "    raise SystemExit(1)\n"
                "print('acceptance passed')\n",
                encoding="utf-8",
            )
            check = _MODULE.grf_guardrail(project)

            rejected, failure = check(None)
            acceptance.write_text("complete", encoding="utf-8")
            accepted, success = check(None)

        self.assertFalse(rejected)
        self.assertIn("incomplete", failure)
        self.assertTrue(accepted)
        self.assertIn("acceptance passed", success)

    def test_missing_cli_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            passed, feedback = _MODULE.grf_guardrail(directory)(None)

        self.assertFalse(passed)
        self.assertIn("grf.py", feedback)

    def test_nonpositive_timeout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "timeout must be greater than zero"):
                _MODULE.grf_guardrail(directory, timeout=0)


if __name__ == "__main__":
    unittest.main()
