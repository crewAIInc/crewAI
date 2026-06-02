"""Tests ensuring the crewai and crewai-cli packages expose consistent entry points.

Regression test for https://github.com/crewAIInc/crewAI/issues/6010:
`uv tool install crewai` failed because only crewai-cli declared [project.scripts].
Both packages must declare the same entry point so that installing either one
via `uv tool install` exposes the `crewai` executable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


LIB_DIR = Path(__file__).resolve().parents[2]
CREWAI_PYPROJECT = LIB_DIR / "crewai" / "pyproject.toml"
CLI_PYPROJECT = LIB_DIR / "cli" / "pyproject.toml"


@pytest.fixture
def crewai_scripts() -> dict[str, str]:
    data = tomllib.loads(CREWAI_PYPROJECT.read_text())
    return data.get("project", {}).get("scripts", {})


@pytest.fixture
def cli_scripts() -> dict[str, str]:
    data = tomllib.loads(CLI_PYPROJECT.read_text())
    return data.get("project", {}).get("scripts", {})


def test_crewai_package_has_crewai_script(crewai_scripts: dict[str, str]) -> None:
    """The crewai package must declare a 'crewai' script entry point."""
    assert "crewai" in crewai_scripts, (
        "lib/crewai/pyproject.toml must have [project.scripts] crewai = ... "
        "so that `uv tool install crewai` exposes the crewai executable."
    )


def test_cli_package_has_crewai_script(cli_scripts: dict[str, str]) -> None:
    """The crewai-cli package must declare a 'crewai' script entry point."""
    assert "crewai" in cli_scripts


def test_entrypoint_targets_same_function(
    crewai_scripts: dict[str, str],
    cli_scripts: dict[str, str],
) -> None:
    """Both packages must point at the same CLI entry function."""
    assert crewai_scripts["crewai"] == cli_scripts["crewai"], (
        "The crewai and crewai-cli packages should declare the same "
        "entry point target for the 'crewai' script."
    )
