"""Regression tests for the MultiOn tool constructor."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from crewai_tools.tools.multion_tool.multion_tool import MultiOnTool


@pytest.fixture
def fake_multion(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    module = types.ModuleType("multion.client")
    module.MultiOn = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "multion", types.ModuleType("multion"))
    monkeypatch.setitem(sys.modules, "multion.client", module)
    return client


def test_constructor_session_id_is_kept(fake_multion: MagicMock) -> None:
    tool = MultiOnTool(api_key="key", session_id="existing-session")

    assert tool.session_id == "existing-session"


def test_run_resumes_the_constructor_session(fake_multion: MagicMock) -> None:
    tool = MultiOnTool(api_key="key", session_id="existing-session")
    tool.multion.browse.return_value = MagicMock(
        session_id="existing-session", message="ok", status="DONE"
    )

    tool._run("open the dashboard")

    assert tool.multion.browse.call_args.kwargs["session_id"] == "existing-session"
