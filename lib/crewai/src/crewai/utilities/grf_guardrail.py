"""Optional local GRF acceptance-check adapter for CrewAI task guardrails.

The adapter invokes the project's existing GRF CLI check and returns the
``(passed, feedback)`` pair expected by ``Task.guardrail``. It never enrolls a
project in cloud services or changes the user's GRF session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from crewai.tasks.task_output import TaskOutput


def grf_guardrail(
    project_dir: str | Path,
    *,
    timeout: float = 120,
) -> Callable[[TaskOutput], tuple[bool, str]]:
    """Create a task guardrail that runs ``grf.py check`` in ``project_dir``.

    The project must already have a GRF workflow/session configured. A failed
    check returns its CLI output as retry feedback; it does not complete the
    task. The command is fixed (no shell), local-only, and bounded by timeout.
    """
    project_path = Path(project_dir).expanduser().resolve()
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    def check(_: TaskOutput) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "grf.py", "check"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"GRF acceptance check could not complete: {exc}"

        feedback = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        if result.returncode:
            return False, feedback or f"GRF acceptance check exited {result.returncode}."
        return True, feedback or "GRF acceptance checks passed."

    return check
