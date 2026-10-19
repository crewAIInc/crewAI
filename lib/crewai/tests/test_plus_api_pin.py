"""`import crewai` pins AMP's proxy and CA before it loads a project's .env.

A classic crew runs in its own process (`uv run run_crew`), which never imports the
CLI. There, `import crewai` is the first thing to read the project's `.env`, so the
machine's transport settings have to be captured before that happens.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from crewai_core import plus_api


def test_import_crewai_pins_the_machines_proxy_before_the_project_env_loads(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "HTTPS_PROXY=http://project-proxy.test:8080\nSSL_CERT_FILE=/project/ca.pem\n"
    )
    env = {key: value for key, value in os.environ.items() if not plus_api._sets_transport(key)}
    probe = (
        "import json, os\n"
        "import crewai\n"
        "from crewai_core import plus_api\n"
        "print(json.dumps({'loaded': os.environ.get('HTTPS_PROXY'),"
        " 'pinned': plus_api._MACHINE_TRANSPORT_ENV}))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    seen = json.loads(result.stdout.strip().splitlines()[-1])
    # crewai loaded the project's .env (so the test means something) ...
    assert seen["loaded"] == "http://project-proxy.test:8080"
    # ... after the pin, which holds only what the machine set: nothing.
    assert seen["pinned"] == {}
