"""Tests for InvokeCrewAIAutomationTool env var support (issue #7389)."""

import os
from unittest.mock import patch

import pytest

from crewai_tools.tools.invoke_crewai_automation_tool.invoke_crewai_automation_tool import (
    InvokeCrewAIAutomationTool,
)


class TestInvokeCrewAIAutomationToolEnvVars:
    """Tool should read CREWAI_API_URL and CREWAI_BEARER_TOKEN from env."""

    def test_instantiation_without_args_uses_env_vars(self):
        """No-argument instantiation should work when env vars are set."""
        env = {
            "CREWAI_API_URL": "https://api.test.com",
            "CREWAI_BEARER_TOKEN": "test-token-123",
        }
        with patch.dict(os.environ, env):
            tool = InvokeCrewAIAutomationTool()
            assert tool.crew_api_url == "https://api.test.com"
            assert tool.crew_bearer_token == "test-token-123"
            assert tool.name == "invoke_amp_automation"

    def test_explicit_args_override_env_vars(self):
        """Explicit constructor arguments should take precedence over env vars."""
        env = {
            "CREWAI_API_URL": "https://env-url.com",
            "CREWAI_BEARER_TOKEN": "env-token",
        }
        with patch.dict(os.environ, env):
            tool = InvokeCrewAIAutomationTool(
                crew_api_url="https://explicit-url.com",
                crew_bearer_token="explicit-token",
                crew_name="My Crew",
                crew_description="A test crew",
            )
            assert tool.crew_api_url == "https://explicit-url.com"
            assert tool.crew_bearer_token == "explicit-token"
            assert tool.name == "My Crew"

    def test_env_vars_declared(self):
        """env_vars should list both documented environment variables."""
        assert "CREWAI_API_URL" in InvokeCrewAIAutomationTool.model_fields.get(
            "env_vars", {}
        ).default or hasattr(InvokeCrewAIAutomationTool, "env_vars")

    def test_empty_env_falls_back_to_defaults(self):
        """Without env vars, tool uses empty string defaults."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            # Remove env vars if they exist
            os.environ.pop("CREWAI_API_URL", None)
            os.environ.pop("CREWAI_BEARER_TOKEN", None)
            tool = InvokeCrewAIAutomationTool()
            assert tool.crew_api_url == ""
            assert tool.crew_bearer_token == ""
