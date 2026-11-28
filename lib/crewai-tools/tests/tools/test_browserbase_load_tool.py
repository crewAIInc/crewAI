import sys
from types import ModuleType
from unittest.mock import MagicMock

from crewai_tools.tools.browserbase_load_tool.browserbase_load_tool import (
    BrowserbaseLoadTool,
)


def test_constructor_credentials_are_used(monkeypatch) -> None:
    """api_key and project_id passed to the constructor must reach the instance."""
    monkeypatch.delenv("BROWSERBASE_API_KEY", raising=False)
    monkeypatch.delenv("BROWSERBASE_PROJECT_ID", raising=False)
    fake_browserbase: ModuleType = ModuleType("browserbase")
    fake_client = MagicMock()
    fake_browserbase.Browserbase = fake_client
    monkeypatch.setitem(sys.modules, "browserbase", fake_browserbase)

    tool = BrowserbaseLoadTool(api_key="constructor-key", project_id="project-1")

    assert tool.api_key == "constructor-key"
    assert tool.project_id == "project-1"
    fake_client.assert_called_once_with(api_key="constructor-key")
