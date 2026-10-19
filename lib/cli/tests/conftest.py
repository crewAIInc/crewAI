"""Shared setup for the CLI's tests."""

import pytest


@pytest.fixture(autouse=True)
def _usage_counts_stay_on_this_machine(monkeypatch):
    """The CLI counts what its commands do (`cli_usage:*`), and a suite that
    exercises them would send those counts as if users had run them: every
    `crewai eval` stop reason within the same second, from a laptop or from CI.
    A test that checks the counting replaces `Telemetry` or sets the flag
    itself."""
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")


@pytest.fixture(autouse=True)
def _each_test_starts_before_any_project_env(monkeypatch):
    """`load_project_env` remembers what a project's `.env` replaced, for the
    life of the process; a test's `.env` must not become the next test's
    "before"."""
    from crewai_cli import run_crew

    monkeypatch.setattr(run_crew, "_ENV_BEFORE_PROJECT", {})
