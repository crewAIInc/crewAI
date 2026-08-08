from unittest.mock import MagicMock, patch

import pytest
from crewai_tools.tools.creduent_verification_tool.creduent_verification_tool import (
    CreduentVerificationSchema,
    CreduentVerificationTool,
)


def test_schema_validation() -> None:
    """Test schema validation for agent URI parameter."""
    schema = CreduentVerificationSchema(agent_uri="agent://assistant.dev/planner")
    assert schema.agent_uri == "agent://assistant.dev/planner"


def test_verification_tool_initialization() -> None:
    """Test tool attributes and default strict configuration."""
    tool = CreduentVerificationTool()
    assert tool.name == "Creduent Agent Identity Verification"
    assert tool.strict is True
    assert tool.package_dependencies == ["creduent"]


@patch("creduent.verify.verify")
def test_successful_verification(mock_verify: MagicMock) -> None:
    """Test successful verification flow when protocol returns valid result."""
    mock_result = MagicMock()
    mock_result.valid = True
    mock_verify.return_value = mock_result

    tool = CreduentVerificationTool(strict=False)
    output = tool._run(agent_uri="agent://assistant.dev/planner")

    assert "Verification SUCCESS" in output
    mock_verify.assert_called_once_with("agent://assistant.dev/planner")


@patch("creduent.verify.verify")
def test_failed_verification_strict(mock_verify: MagicMock) -> None:
    """Test verification failure when strict mode is active."""
    mock_result = MagicMock()
    mock_result.valid = False
    mock_result.error = "Invalid signature"
    mock_verify.return_value = mock_result

    tool = CreduentVerificationTool(strict=True)

    with pytest.raises(ValueError) as exc_info:
        tool._run(agent_uri="agent://untrusted.dev/hacker")

    assert str(exc_info.value) == "Verification FAILED for agent://untrusted.dev/hacker: Invalid signature"


@patch("creduent.verify.verify")
def test_unexpected_exception_handling(mock_verify: MagicMock) -> None:
    """Test handling of unexpected exceptions raised during verification."""
    mock_verify.side_effect = RuntimeError("Network timeout")

    tool = CreduentVerificationTool(strict=True)

    with pytest.raises(ValueError) as exc_info:
        tool._run(agent_uri="agent://untrusted.dev/hacker")

    assert "Verification failure for agent://untrusted.dev/hacker: Network timeout" in str(exc_info.value)


def test_missing_package_import_error_strict() -> None:
    """Test missing creduent package handling when strict mode is active."""
    tool = CreduentVerificationTool(strict=True)

    with patch.dict("sys.modules", {"creduent": None, "creduent.verify": None}):
        with pytest.raises(ValueError) as exc_info:
            tool._run(agent_uri="agent://assistant.dev/planner")

    assert "Error: creduent package is not installed." in str(exc_info.value)


def test_missing_package_import_error_non_strict() -> None:
    """Test missing creduent package handling when strict mode is disabled."""
    tool = CreduentVerificationTool(strict=False)

    with patch.dict("sys.modules", {"creduent": None, "creduent.verify": None}):
        output = tool._run(agent_uri="agent://assistant.dev/planner")

    assert "Error: creduent package is not installed." in output


