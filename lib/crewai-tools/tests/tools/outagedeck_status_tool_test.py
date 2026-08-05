import json
from unittest.mock import Mock, patch

import pytest
import requests

from crewai_tools.tools.outagedeck_status_tool import OutageDeckStatusTool


@pytest.fixture
def tool() -> OutageDeckStatusTool:
    return OutageDeckStatusTool(timeout=7)


def response(payload: object) -> Mock:
    result = Mock()
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def incident() -> dict[str, object]:
    return {
        "slug": "github-actions-delays-2026-08-05",
        "title": "Actions delays",
        "summary": "Some workflow runs are delayed.",
        "status": "monitoring",
        "severity": "major",
        "startedAt": "2026-08-05T10:00:00Z",
        "updatedAt": "2026-08-05T11:00:00Z",
        "resolvedAt": None,
        "provider": {"slug": "github", "name": "GitHub"},
        "affectedServices": [{"slug": "github-actions", "name": "GitHub Actions"}],
    }


def provider_payload() -> dict[str, object]:
    return {
        "data": {
            "slug": "github",
            "name": "GitHub",
            "currentStatus": {
                "code": "degraded",
                "label": "Degraded Performance",
                "headline": "Some systems are degraded",
                "summary": "GitHub reports degraded performance.",
                "capturedAt": "2026-08-05T12:00:00Z",
            },
            "counts": {"services": 3, "activeIncidents": 1, "incidents": 10},
            "services": [
                {
                    "slug": "github-actions",
                    "name": "GitHub Actions",
                    "category": "ci-cd",
                    "status": "degraded",
                    "summary": "Workflow execution.",
                }
            ],
            "activeIncidents": [incident()],
        }
    }


def service_payload() -> dict[str, object]:
    return {
        "data": {
            "slug": "github-actions",
            "name": "GitHub Actions",
            "category": "ci-cd",
            "status": "degraded",
            "summary": "Workflow execution.",
            "provider": {"slug": "github", "name": "GitHub"},
            "counts": {"incidents": 12, "activeIncidents": 1},
            "incidents": [incident()] * 12,
        }
    }


def parsed(result: str) -> dict[str, object]:
    value = json.loads(result)
    assert isinstance(value, dict)
    return value


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_provider_status_request_and_response(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.return_value = response(provider_payload())

    result = parsed(tool.run(operation="provider_status", slug=" GitHub "))

    mock_get.assert_called_once_with(
        "https://outagedeck.com/api/v1/providers/github",
        params={},
        headers={"Accept": "application/json", "User-Agent": "CrewAI-OutageDeck/1.0"},
        timeout=7,
        max_redirects=0,
    )
    assert result["success"] is True
    assert result["provider"] == {"slug": "github", "name": "GitHub"}
    assert result["status"]["code"] == "degraded"
    assert result["active_incidents"][0]["severity"] == "major"
    assert "utm_campaign=crewai_tool" in result["outagedeck_url"]


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_incident_filters_and_pagination(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.return_value = response(
        {
            "data": {
                "count": 1,
                "page": 2,
                "totalPages": 4,
                "totalIncidents": 7,
                "incidents": [incident()],
            }
        }
    )

    result = parsed(
        tool.run(
            operation="list_incidents",
            provider=" GitHub ",
            state="active",
            severity="major",
            page=2,
            limit=5,
        )
    )

    mock_get.assert_called_once_with(
        "https://outagedeck.com/api/v1/incidents",
        params={
            "page": 2,
            "limit": 5,
            "provider": "github",
            "state": "active",
            "severity": "major",
        },
        headers={"Accept": "application/json", "User-Agent": "CrewAI-OutageDeck/1.0"},
        timeout=7,
        max_redirects=0,
    )
    assert result["total_pages"] == 4
    assert result["incidents"][0]["affected_services"][0]["slug"] == "github-actions"
    assert "utm_source=crewai" in result["incidents"][0]["outagedeck_url"]


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_service_status_caps_incident_context(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.return_value = response(service_payload())

    result = parsed(tool.run(operation="service_status", slug="github-actions"))

    assert result["service"]["status"] == "degraded"
    assert result["provider"] == {"slug": "github", "name": "GitHub"}
    assert len(result["recent_incidents"]) == 10
    assert "utm_medium=integration" in result["outagedeck_url"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operation": "provider_status"}, "slug is required"),
        ({"operation": "service_status", "slug": "../admin"}, "slugs must contain"),
        ({"operation": "provider_status", "slug": "github/status"}, "slugs must contain"),
        ({"operation": "list_incidents", "provider": "github?limit=100"}, "slugs must contain"),
        ({"operation": "list_incidents", "page": 0}, "greater than or equal to 1"),
        ({"operation": "list_incidents", "limit": 101}, "less than or equal to 100"),
        ({"operation": "list_incidents", "state": "investigating"}, "active"),
        ({"operation": "list_incidents", "severity": "catastrophic"}, "minor"),
    ],
)
@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_invalid_input_never_requests(
    mock_get: Mock,
    kwargs: dict[str, object],
    message: str,
    tool: OutageDeckStatusTool,
) -> None:
    with pytest.raises(ValueError, match=message):
        tool.run(**kwargs)

    mock_get.assert_not_called()


@pytest.mark.parametrize(
    ("status", "message"),
    [(404, "could not find"), (429, "rate limit"), (500, "HTTP 500")],
)
@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_http_errors_return_stable_failure(
    mock_get: Mock,
    status: int,
    message: str,
    tool: OutageDeckStatusTool,
) -> None:
    http_response = requests.Response()
    http_response.status_code = status
    http_response.url = "https://outagedeck.com/api/v1/providers/missing"
    mock_get.return_value = http_response

    result = parsed(tool.run(operation="provider_status", slug="missing"))

    assert result["success"] is False
    assert message.lower() in result["error"].lower()


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_network_error_returns_stable_failure(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.side_effect = requests.ConnectTimeout("timed out")

    result = parsed(tool.run(operation="list_incidents"))

    assert result == {"success": False, "error": "Could not reach OutageDeck: timed out"}


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_redirect_is_rejected(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.side_effect = ValueError("Too many redirects")

    result = parsed(tool.run(operation="list_incidents"))

    assert result == {
        "success": False,
        "error": "OutageDeck request rejected: Too many redirects",
    }


@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_invalid_json_returns_stable_failure(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_response = response({})
    mock_response.json.side_effect = ValueError("invalid document")
    mock_get.return_value = mock_response

    result = parsed(tool.run(operation="provider_status", slug="github"))

    assert result == {
        "success": False,
        "error": "OutageDeck returned invalid JSON: invalid document",
    }


@pytest.mark.parametrize(
    ("operation", "slug"),
    [
        ("provider_status", "github"),
        ("list_incidents", None),
        ("service_status", "github-actions"),
    ],
)
@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
def test_unexpected_response_shape_is_actionable(
    mock_get: Mock,
    operation: str,
    slug: str | None,
    tool: OutageDeckStatusTool,
) -> None:
    mock_get.return_value = response({"data": []})

    result = parsed(tool.run(operation=operation, slug=slug))

    assert result["success"] is False
    assert "unexpected" in result["error"]


@pytest.mark.asyncio
@patch("crewai_tools.tools.outagedeck_status_tool.outagedeck_status_tool.safe_get")
async def test_async_execution_matches_sync(mock_get: Mock, tool: OutageDeckStatusTool) -> None:
    mock_get.return_value = response(provider_payload())

    result = parsed(await tool.arun(operation="provider_status", slug="github"))

    assert result["success"] is True
    assert result["provider"] == {"slug": "github", "name": "GitHub"}


def test_exported_from_package() -> None:
    from crewai_tools import OutageDeckStatusTool as ExportedTool
    from crewai_tools.tools import OutageDeckStatusTool as ToolsExportedTool

    assert ExportedTool is OutageDeckStatusTool
    assert ToolsExportedTool is OutageDeckStatusTool
