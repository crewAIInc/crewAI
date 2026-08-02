from unittest.mock import MagicMock, patch
import pytest

from crewai_tools.tools.creduent_verification_tool.creduent_verification_tool import (
    CreduentVerificationSchema,
    CreduentVerificationTool,
)


def test_schema_validation():
    schema = CreduentVerificationSchema(agent_uri="agent://assistant.dev/planner")
    assert schema.agent_uri == "agent://assistant.dev/planner"


def test_verification_tool_initialization():
    tool = CreduentVerificationTool()
    assert tool.name == "Creduent Agent Identity Verification"
    assert tool.strict is True


@patch("creduent.verify.verify")
def test_successful_verification(mock_verify):
    mock_result = MagicMock()
    mock_result.valid = True
    mock_verify.return_value = mock_result

    tool = CreduentVerificationTool(strict=False)
    output = tool._run(agent_uri="agent://assistant.dev/planner")

    assert "Verification SUCCESS" in output
    mock_verify.assert_called_once_with("agent://assistant.dev/planner")


@patch("creduent.verify.verify")
def test_failed_verification_strict(mock_verify):
    mock_result = MagicMock()
    mock_result.valid = False
    mock_result.error = "Invalid signature"
    mock_verify.return_value = mock_result

    tool = CreduentVerificationTool(strict=True)

    with pytest.raises(ValueError) as exc_info:
        tool._run(agent_uri="agent://untrusted.dev/hacker")

    assert "Invalid signature" in str(exc_info.value)
