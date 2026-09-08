"""Tests for the repository's local pre-commit hooks."""

from pathlib import Path

import yaml


def test_local_hooks_use_platform_specific_activation() -> None:
    """Local hooks should activate the correct virtual environment path."""
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )

    local_hooks = [
        hook
        for repository in config["repos"]
        if repository["repo"] == "local"
        for hook in repository["hooks"]
        if "entry" in hook
    ]

    assert local_hooks
    assert all(hook["entry"].startswith("bash -c ") for hook in local_hooks)
    assert all("$OSTYPE" in hook["entry"] for hook in local_hooks)
    assert all("source .venv/bin/activate" in hook["entry"] for hook in local_hooks)
    assert all("source .venv/Scripts/activate" in hook["entry"] for hook in local_hooks)

    expected_prefixes = {
        "ruff": "uv run ruff check --config pyproject.toml",
        "ruff-format": "uv run ruff format --config pyproject.toml",
        "mypy": "uv run mypy --config-file pyproject.toml",
        "pip-audit": "uv run pip-audit --skip-editable",
    }
    hooks_by_id = {hook["id"]: hook for hook in local_hooks}
    assert set(expected_prefixes) <= set(hooks_by_id)
    for hook_id, prefix in expected_prefixes.items():
        assert prefix in hooks_by_id[hook_id]["entry"]
