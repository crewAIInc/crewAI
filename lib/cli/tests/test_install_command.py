"""Regression tests for the `crewai install` exit status."""

from __future__ import annotations

import subprocess
from typing import Any

from click.testing import CliRunner
import pytest

from crewai_cli import install_crew as install_module
from crewai_cli.cli import crewai


def test_install_exits_with_the_uv_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A failed `uv sync` must not be reported as a successful install."""

    def failing_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(
            2, ["uv", "sync"], output="No solution found"
        )

    monkeypatch.setattr(install_module.subprocess, "run", failing_run)
    monkeypatch.setattr(
        install_module, "build_env_with_all_tool_credentials", lambda: {}
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')

    result = CliRunner().invoke(crewai, ["install"])

    assert result.exit_code == 2, result.output
    assert "An error occurred while running the crew" in result.output
