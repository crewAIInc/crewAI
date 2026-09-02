from unittest.mock import MagicMock, patch

from crewai_tools.tools.agenttoll_safety_tool.agenttoll_safety_tool import (
    AgentTollSafetyTool,
)
import pytest


DUMMY_KEY = "0x" + "11" * 32  # syntactically valid, never funded


@pytest.fixture
def agenttoll_tool(monkeypatch):
    monkeypatch.setenv("EVM_PRIVATE_KEY", DUMMY_KEY)
    return AgentTollSafetyTool()


def test_requires_env_var(monkeypatch):
    monkeypatch.delenv("EVM_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError):
        AgentTollSafetyTool()


def test_happy_path(agenttoll_tool):
    mock_response = MagicMock()
    mock_response.text = '{"verdict": "clear"}'
    mock_response.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = False
    mock_session.get.return_value = mock_response

    with patch.object(AgentTollSafetyTool, "_paid_session", return_value=mock_session):
        result = agenttoll_tool.run(address="0x940181a94a35a4569e4529a3cdfb74e38fd98631")

    assert "clear" in result
    mock_session.get.assert_called_once_with(
        "https://agenttoll.app/api/base/safety/0x940181a94a35a4569e4529a3cdfb74e38fd98631"
    )


def test_error_is_returned_not_raised(agenttoll_tool):
    with patch.object(AgentTollSafetyTool, "_paid_session", side_effect=RuntimeError("boom")):
        result = agenttoll_tool.run(address="0x940181a94a35a4569e4529a3cdfb74e38fd98631")

    assert "Error checking token safety" in result
    assert "boom" in result
