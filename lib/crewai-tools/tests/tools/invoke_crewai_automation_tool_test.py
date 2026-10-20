import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.invoke_crewai_automation_tool.invoke_crewai_automation_tool import (
    DEFAULT_TOOL_DESCRIPTION,
    DEFAULT_TOOL_NAME,
    InvokeCrewAIAutomationTool,
)
from pydantic import Field
import pytest


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure CREWAI_API_URL/CREWAI_BEARER_TOKEN never leak between tests."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CREWAI_API_URL", None)
        os.environ.pop("CREWAI_BEARER_TOKEN", None)
        yield


def test_zero_argument_instantiation_does_not_raise():
    """A runtime that resolves tools by class reference (e.g. CrewAI AMP Studio's
    "Invoke Amp Automation" internal tool) instantiates the tool with no arguments.
    This must never raise a TypeError about missing positional arguments."""
    tool = InvokeCrewAIAutomationTool()

    assert tool.name == DEFAULT_TOOL_NAME
    assert DEFAULT_TOOL_DESCRIPTION in tool.description
    assert tool.crew_api_url is None
    assert tool.crew_bearer_token is None


def test_existing_positional_keyword_call_shape_still_works():
    """Backward compatibility: the documented/previous required-arguments call shape
    keeps working unchanged."""
    tool = InvokeCrewAIAutomationTool(
        "https://api.example.com",
        "explicit_token",
        "My Crew",
        "Description of what the crew does",
    )

    assert tool.crew_api_url == "https://api.example.com"
    assert tool.crew_bearer_token == "explicit_token"
    assert tool.name == "My Crew"
    assert "Description of what the crew does" in tool.description

    kwarg_tool = InvokeCrewAIAutomationTool(
        crew_api_url="https://api.example.com",
        crew_bearer_token="explicit_token",
        crew_name="My Crew",
        crew_description="Description of what the crew does",
        max_polling_time=120,
    )
    assert kwarg_tool.crew_api_url == "https://api.example.com"
    assert kwarg_tool.crew_bearer_token == "explicit_token"
    assert kwarg_tool.max_polling_time == 120


def test_env_var_fallback_for_url_and_token():
    """Omitted url/token are read from CREWAI_API_URL/CREWAI_BEARER_TOKEN."""
    with patch.dict(
        os.environ,
        {
            "CREWAI_API_URL": "https://from-env.crewai.com",
            "CREWAI_BEARER_TOKEN": "env_token",
        },
    ):
        tool = InvokeCrewAIAutomationTool(
            crew_name="My Crew", crew_description="Does things"
        )

    assert tool.crew_api_url == "https://from-env.crewai.com"
    assert tool.crew_bearer_token == "env_token"
    assert tool.name == "My Crew"
    assert "Does things" in tool.description


def test_explicit_arguments_take_precedence_over_env_vars():
    """An explicitly passed url/token wins over whatever the environment holds."""
    with patch.dict(
        os.environ,
        {
            "CREWAI_API_URL": "https://from-env.crewai.com",
            "CREWAI_BEARER_TOKEN": "env_token",
        },
    ):
        tool = InvokeCrewAIAutomationTool(
            crew_api_url="https://explicit.crewai.com",
            crew_bearer_token="explicit_token",
            crew_name="My Crew",
            crew_description="Does things",
        )

    assert tool.crew_api_url == "https://explicit.crewai.com"
    assert tool.crew_bearer_token == "explicit_token"


def test_missing_configuration_raises_clear_error_on_use():
    """No explicit args and no env vars: construction succeeds, but running the
    tool must raise a clear, actionable error instead of failing deep inside
    `requests` or the CrewAI Platform API with a bare 401."""
    tool = InvokeCrewAIAutomationTool(
        crew_name="My Crew", crew_description="Does things"
    )

    with pytest.raises(ValueError) as exc_info:
        tool.run(prompt="hello")

    message = str(exc_info.value)
    assert "CREWAI_API_URL" in message
    assert "CREWAI_BEARER_TOKEN" in message


def test_partial_configuration_raises_clear_error_on_use():
    """Only one of the two required values is configured."""
    tool = InvokeCrewAIAutomationTool(
        crew_api_url="https://api.example.com",
        crew_name="My Crew",
        crew_description="Does things",
    )

    with pytest.raises(ValueError) as exc_info:
        tool.run(prompt="hello")

    message = str(exc_info.value)
    assert "CREWAI_BEARER_TOKEN" in message
    assert "CREWAI_API_URL" not in message


@patch("requests.get")
@patch("requests.post")
def test_successful_run_with_env_var_configuration(mock_post, mock_get):
    """End-to-end run (mocked HTTP) using only environment-provided credentials."""
    with patch.dict(
        os.environ,
        {
            "CREWAI_API_URL": "https://from-env.crewai.com",
            "CREWAI_BEARER_TOKEN": "env_token",
        },
    ):
        tool = InvokeCrewAIAutomationTool(
            crew_name="My Crew", crew_description="Does things"
        )

    kickoff_response = MagicMock()
    kickoff_response.json.return_value = {"kickoff_id": "kickoff-123"}
    mock_post.return_value = kickoff_response

    status_response = MagicMock()
    status_response.json.return_value = {"state": "success", "result": "42"}
    mock_get.return_value = status_response

    result = tool.run(prompt="hello")

    assert result == "42"
    mock_post.assert_called_once_with(
        "https://from-env.crewai.com/kickoff",
        headers={
            "Authorization": "Bearer env_token",
            "Content-Type": "application/json",
        },
        json={"inputs": {"prompt": "hello"}},
        timeout=30,
    )


def test_dynamic_crew_inputs_schema_still_works():
    """crew_inputs keeps building a dynamic args schema for the automation inputs."""
    custom_inputs = {
        "year": Field(..., description="Year to retrieve the report for (integer)"),
        "region": Field(default="global", description="Geographic region"),
    }

    tool = InvokeCrewAIAutomationTool(
        crew_api_url="https://api.example.com",
        crew_bearer_token="token",
        crew_name="State of AI Report",
        crew_description="Retrieves a report on state of AI for a given year.",
        crew_inputs=custom_inputs,
    )

    schema_fields = tool.args_schema.model_fields
    assert "year" in schema_fields
    assert "region" in schema_fields


def test_explicit_empty_value_is_not_replaced_by_env_var():
    """An explicit empty string is a caller mistake, not a request to read the
    environment: it must reach _ensure_configured() and be reported as missing."""
    with patch.dict(
        os.environ,
        {
            "CREWAI_API_URL": "https://from-env.crewai.com",
            "CREWAI_BEARER_TOKEN": "env_token",
        },
    ):
        tool = InvokeCrewAIAutomationTool(
            crew_api_url="",
            crew_name="My Crew",
            crew_description="Does things",
        )

    assert tool.crew_api_url == ""
    assert tool.crew_bearer_token == "env_token"
    with pytest.raises(ValueError) as exc_info:
        tool.run(prompt="hello")

    message = str(exc_info.value)
    assert "CREWAI_API_URL" in message
    assert "CREWAI_BEARER_TOKEN" not in message
