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


@pytest.fixture
def no_provider_keys(monkeypatch):
    """Unset every value `crewai create` prompts for, so a key exported on the
    machine running the suite does not stop the CLI from naming it as missing."""
    from crewai_cli.constants import ENV_VARS

    for entries in ENV_VARS.values():
        for details in entries:
            if "key_name" in details:
                monkeypatch.delenv(str(details["key_name"]).upper(), raising=False)


@pytest.fixture(autouse=True)
def _no_coding_agent_detected(monkeypatch):
    """Run as CI does, whoever runs the suite: an assistant's markers in the
    environment make a command with no login create a temporary AMP account
    (`crewai_cli.authentication.placeholder`). A test that needs an assistant
    sets its marker itself."""
    from crewai_core.runtime_env import (
        CODING_AGENT_ENV_MARKERS,
        GENERIC_AGENT_ENV_VARS,
    )

    for _, names in CODING_AGENT_ENV_MARKERS:
        for name in names:
            monkeypatch.delenv(name, raising=False)
    for name in GENERIC_AGENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
