"""`import crewai` must not load a project's `.env` when opted out.

Fixes #9744: `crewai.llm` and `crewai.project.crew_base` both call
`load_dotenv()` at import time with no gate, so a bare `import crewai` from
any directory mutates `os.environ` for the whole process. Setting
`CREWAI_NO_AUTO_DOTENV` to a truthy value opts out; the default (unset)
keeps the historical autoload behavior.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_probe(tmp_path: Path, flag: str | None) -> dict[str, str | None]:
    (tmp_path / ".env").write_text("CREWAI_PROBE_MARKER=from-project-dotenv\n")
    env = dict(os.environ)
    env.pop("CREWAI_NO_AUTO_DOTENV", None)
    env.pop("CREWAI_PROBE_MARKER", None)
    if flag is not None:
        env["CREWAI_NO_AUTO_DOTENV"] = flag
    probe = (
        "import json, os\n"
        "import crewai\n"
        "import crewai.project.crew_base\n"
        "print(json.dumps({'marker': os.environ.get('CREWAI_PROBE_MARKER')}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on", "TRUE", " True "])
def test_opt_out_flag_stops_import_time_dotenv_loading(
    tmp_path: Path, flag: str
) -> None:
    seen = _run_probe(tmp_path, flag)
    assert seen["marker"] is None


@pytest.mark.parametrize("flag", [None, "", "0", "false", "no"])
def test_default_and_falsy_flag_keep_import_time_dotenv_loading(
    tmp_path: Path, flag: str | None
) -> None:
    seen = _run_probe(tmp_path, flag)
    assert seen["marker"] == "from-project-dotenv"
