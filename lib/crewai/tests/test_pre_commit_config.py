"""Tests for the repository's local pre-commit hooks."""

from pathlib import Path

import yaml


def test_local_hooks_use_uv_without_shell_activation() -> None:
    """Local hooks should work on Windows as well as Unix-like systems."""
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
    assert all(hook["entry"].startswith("uv run ") for hook in local_hooks)
    assert all(".venv/bin/activate" not in hook["entry"] for hook in local_hooks)
