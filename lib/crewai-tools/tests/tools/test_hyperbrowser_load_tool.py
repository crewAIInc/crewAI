import sys
from types import ModuleType
from unittest.mock import MagicMock

from crewai_tools.tools.hyperbrowser_load_tool.hyperbrowser_load_tool import (
    HyperbrowserLoadTool,
)


def test_environment_api_key_is_accepted(monkeypatch) -> None:
    """HYPERBROWSER_API_KEY alone must be enough to construct the tool."""
    monkeypatch.setenv("HYPERBROWSER_API_KEY", "env-key")
    fake_hyperbrowser: ModuleType = ModuleType("hyperbrowser")
    fake_client = MagicMock()
    fake_hyperbrowser.Hyperbrowser = fake_client
    monkeypatch.setitem(sys.modules, "hyperbrowser", fake_hyperbrowser)

    tool = HyperbrowserLoadTool()

    assert tool.api_key == "env-key"
    fake_client.assert_called_once_with(api_key="env-key")
