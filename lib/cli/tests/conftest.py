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
