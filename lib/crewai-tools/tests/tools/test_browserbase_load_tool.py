import sys
from unittest.mock import MagicMock, patch

from crewai_tools.tools.browserbase_load_tool.browserbase_load_tool import (
    BrowserbaseLoadTool,
)


def _build_tool(monkeypatch, **credentials):
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)
    fake_browserbase_module = MagicMock()
    with patch.dict(sys.modules, {"browserbase": fake_browserbase_module}):
        tool = BrowserbaseLoadTool(**credentials)
    return tool, fake_browserbase_module


def test_explicit_credentials_honored_when_env_missing(monkeypatch):
    tool, fake_browserbase_module = _build_tool(
        monkeypatch, api_key="bb-test-key", project_id="proj-1"
    )

    assert tool.api_key == "bb-test-key"
    assert tool.project_id == "proj-1"
    fake_browserbase_module.Browserbase.assert_called_once_with(api_key="bb-test-key")


def test_explicit_credentials_override_environment(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "env-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "env-project")
    fake_browserbase_module = MagicMock()
    with patch.dict(sys.modules, {"browserbase": fake_browserbase_module}):
        tool = BrowserbaseLoadTool(api_key="bb-test-key", project_id="proj-1")

    assert tool.api_key == "bb-test-key"
    assert tool.project_id == "proj-1"


def test_environment_fallback_when_no_credentials_passed(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "env-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "env-project")
    fake_browserbase_module = MagicMock()
    with patch.dict(sys.modules, {"browserbase": fake_browserbase_module}):
        tool = BrowserbaseLoadTool()

    assert tool.api_key == "env-key"
    assert tool.project_id == "env-project"
