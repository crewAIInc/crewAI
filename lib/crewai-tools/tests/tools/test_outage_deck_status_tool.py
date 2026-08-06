"""Tests for OutageDeckStatusTool."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from crewai_tools import OutageDeckStatusTool
import pytest


class TestOutageDeckStatusTool:
    """Tests for OutageDeckStatusTool."""

    def test_tool_is_base_tool_subclass(self):
        """Test that OutageDeckStatusTool is a proper BaseTool subclass."""
        tool = OutageDeckStatusTool()
        assert tool.name == "OutageDeck status check"
        assert "OutageDeck" in tool.description
        assert tool.args_schema is not None

    def test_provider_status_builds_correct_url(self):
        """Test _build_request for provider_status action."""
        url, params = OutageDeckStatusTool._build_request(
            action="provider_status",
            provider_slug="aws",
            service_slug=None,
            lifecycle=None,
            severity=None,
            page=1,
        )
        assert url == "https://outagedeck.com/api/v1/providers/aws"
        assert params == {}

    def test_service_status_builds_correct_url(self):
        """Test _build_request for service_status action."""
        url, params = OutageDeckStatusTool._build_request(
            action="service_status",
            provider_slug=None,
            service_slug="aws-ec2",
            lifecycle=None,
            severity=None,
            page=1,
        )
        assert url == "https://outagedeck.com/api/v1/services/aws-ec2"
        assert params == {}

    def test_list_incidents_builds_correct_url(self):
        """Test _build_request for list_incidents action with no filters."""
        url, params = OutageDeckStatusTool._build_request(
            action="list_incidents",
            provider_slug=None,
            service_slug=None,
            lifecycle=None,
            severity=None,
            page=1,
        )
        assert url == "https://outagedeck.com/api/v1/incidents"
        assert params == {}

    def test_list_incidents_with_all_filters(self):
        """Test _build_request for list_incidents with all filters."""
        url, params = OutageDeckStatusTool._build_request(
            action="list_incidents",
            provider_slug="github",
            service_slug=None,
            lifecycle="active",
            severity="critical",
            page=3,
        )
        assert url == "https://outagedeck.com/api/v1/incidents"
        assert params == {
            "provider": "github",
            "lifecycle": "active",
            "severity": "critical",
            "page": 3,
        }

    def test_list_incidents_page_1_not_included(self):
        """Test that page=1 is not sent as a query param (default)."""
        _, params = OutageDeckStatusTool._build_request(
            action="list_incidents",
            provider_slug=None,
            service_slug=None,
            lifecycle=None,
            severity=None,
            page=1,
        )
        assert "page" not in params

    def test_provider_status_requires_provider_slug(self):
        """Test that provider_status raises ValueError without provider_slug."""
        with pytest.raises(ValueError, match="provider_slug is required"):
            OutageDeckStatusTool._build_request(
                action="provider_status",
                provider_slug=None,
                service_slug=None,
                lifecycle=None,
                severity=None,
                page=1,
            )

    def test_service_status_requires_service_slug(self):
        """Test that service_status raises ValueError without service_slug."""
        with pytest.raises(ValueError, match="service_slug is required"):
            OutageDeckStatusTool._build_request(
                action="service_status",
                provider_slug=None,
                service_slug=None,
                lifecycle=None,
                severity=None,
                page=1,
            )

    def test_invalid_action_raises(self):
        """Test that an unknown action raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            OutageDeckStatusTool._build_request(
                action="invalid_action",
                provider_slug=None,
                service_slug=None,
                lifecycle=None,
                severity=None,
                page=1,
            )

    def test_invalid_provider_slug_format_raises(self):
        """Test that provider_slug with invalid format is rejected by schema."""
        from crewai_tools.tools.outage_deck_status_tool.outage_deck_status_tool import (
            OutageDeckStatusToolSchema,
        )

        with pytest.raises(ValueError):
            OutageDeckStatusToolSchema(
                action="provider_status", provider_slug="INVALID_SLUG!"
            )

    def test_invalid_lifecycle_raises(self):
        """Test that invalid lifecycle value is rejected."""
        from crewai_tools.tools.outage_deck_status_tool.outage_deck_status_tool import (
            OutageDeckStatusToolSchema,
        )

        with pytest.raises(ValueError, match="lifecycle must be one of"):
            OutageDeckStatusToolSchema(
                action="list_incidents", lifecycle="invalid"
            )

    def test_invalid_severity_raises(self):
        """Test that invalid severity value is rejected."""
        from crewai_tools.tools.outage_deck_status_tool.outage_deck_status_tool import (
            OutageDeckStatusToolSchema,
        )

        with pytest.raises(ValueError, match="severity must be one of"):
            OutageDeckStatusToolSchema(
                action="list_incidents", severity="invalid"
            )

    def test_run_returns_json_string(self):
        """Test that _run returns a JSON string with the API response."""
        tool = OutageDeckStatusTool()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"status": "operational", "slug": "vercel"}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            result = tool._run(action="provider_status", provider_slug="vercel")

            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == "https://outagedeck.com/api/v1/providers/vercel"
            assert kwargs["timeout"] == 15.0

            parsed = json.loads(result)
            assert parsed["data"]["status"] == "operational"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Async tests require an event loop, which uses internal socketpair() "
        "that is blocked by pytest-recording's --block-network on Windows. "
        "CI runs on Linux and covers async execution paths.",
    )
    async def test_arun_returns_json_string(self):
        """Test that _arun returns a JSON string with the API response."""
        tool = OutageDeckStatusTool()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"status": "operational", "slug": "vercel"}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool._arun(
                action="provider_status", provider_slug="vercel"
            )

            mock_client.get.assert_called_once()
            parsed = json.loads(result)
            assert parsed["data"]["slug"] == "vercel"
