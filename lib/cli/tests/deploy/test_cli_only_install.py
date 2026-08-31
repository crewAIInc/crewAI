"""Regression coverage for crewai-cli installed without the full crewai package."""

import builtins
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

import crewai_cli.deploy.validate as validate_module


def test_deploy_and_install_modules_import_without_crewai() -> None:
    script = """
import builtins

real_import = builtins.__import__

def import_without_crewai(name, *args, **kwargs):
    if name == "crewai" or name.startswith("crewai."):
        raise ModuleNotFoundError("No module named 'crewai'", name="crewai")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_crewai

import crewai_cli.deploy.main
import crewai_cli.install_crew
"""

    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


def test_json_validation_uses_project_environment_without_crewai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_import = builtins.__import__

    def import_without_crewai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "crewai" or name.startswith("crewai."):
            raise ModuleNotFoundError("No module named 'crewai'", name="crewai")
        return real_import(name, *args, **kwargs)

    captured: dict[str, Any] = {}

    def fake_run(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        payload = {"ok": True, "agent_names": ["researcher"]}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "uv output\n"
                f"{validate_module._JSON_VALIDATION_MARKER}{json.dumps(payload)}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(builtins, "__import__", import_without_crewai)
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(subprocess, "run", fake_run)

    crew_path = tmp_path / "crew.jsonc"
    agents_dir = tmp_path / "agents"
    assert validate_module._validate_json_project(
        crew_path, agents_dir, tmp_path
    ) == ["researcher"]

    assert captured["command"][:4] == ["/usr/bin/uv", "run", "python", "-c"]
    assert captured["kwargs"]["cwd"] == tmp_path


def test_project_environment_preserves_json_validation_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {"ok": False, "errors": ["tasks[0] references missing_agent"]}
    proc = subprocess.CompletedProcess(
        [],
        0,
        stdout=f"{validate_module._JSON_VALIDATION_MARKER}{json.dumps(payload)}\n",
        stderr="",
    )
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: proc)

    with pytest.raises(validate_module._JSONProjectValidationError) as exc_info:
        validate_module._validate_json_project_in_project_env(
            tmp_path / "crew.jsonc", tmp_path / "agents", tmp_path
        )

    assert exc_info.value.errors == ["tasks[0] references missing_agent"]
