"""Tests ensuring the crewai package exposes the CLI entry point.

Regression test for https://github.com/crewAIInc/crewAI/issues/6010:
`uv tool install crewai` failed because the crewai package did not declare
any [project.scripts], so uv could not find an executable to expose.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import click
import pytest
import tomllib


CREWAI_PYPROJECT = (
    Path(__file__).resolve().parents[2] / "pyproject.toml"
)


@pytest.fixture
def crewai_metadata() -> dict:
    """Load the crewai package pyproject.toml as a dict."""
    return tomllib.loads(CREWAI_PYPROJECT.read_text())


def test_crewai_package_declares_scripts_entrypoint(crewai_metadata: dict) -> None:
    """The crewai package must declare a 'crewai' console script."""
    scripts = crewai_metadata.get("project", {}).get("scripts", {})
    assert "crewai" in scripts, (
        "The crewai package pyproject.toml must define [project.scripts] "
        "with a 'crewai' entry so that `uv tool install crewai` works."
    )


def test_crewai_entrypoint_target_is_importable(crewai_metadata: dict) -> None:
    """The target of the crewai script entry point must be importable."""
    scripts = crewai_metadata.get("project", {}).get("scripts", {})
    ref = scripts.get("crewai", "")
    assert ":" in ref, f"Entry point reference should be 'module:attr', got: {ref!r}"
    module_path, attr_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    entry = getattr(mod, attr_name, None)
    assert entry is not None, (
        f"Could not find attribute {attr_name!r} in module {module_path!r}"
    )


def test_crewai_entrypoint_is_click_command(crewai_metadata: dict) -> None:
    """The crewai CLI entry point must be a click command/group."""
    scripts = crewai_metadata.get("project", {}).get("scripts", {})
    ref = scripts["crewai"]
    module_path, attr_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    entry = getattr(mod, attr_name)
    assert isinstance(entry, click.BaseCommand), (
        f"Expected a click command/group, got {type(entry).__name__}"
    )
