import json
from unittest.mock import MagicMock, patch

import pytest
from getanyapi import (
    AnyAPIError,
    Balance,
    CatalogEntry,
    CatalogSearchResults,
    InsufficientBalanceError,
    RunResult,
)
from pydantic import ValidationError

from crewai_tools.tools.anyapi_tool import (
    AnyApiDescribeTool,
    AnyApiRunTool,
    AnyApiSearchTool,
)
from crewai_tools.tools.anyapi_tool.anyapi_run import AnyApiRunToolSchema

SEARCH_RESULT = {
    "slug": "instagram.profile",
    "platformId": "instagram",
    "name": "Instagram Profile",
    "description": "Fetch an Instagram account's public profile by handle.",
    "category": "social",
    "method": "POST",
    "path": "/v1/run/instagram.profile",
    "execution": {"mode": "sync"},
    "provider": "AnyAPI",
    "pricing": {
        "from": {
            "model": "flat",
            "unit": "request",
            "maxUsd": 0.002,
            "maxPer1kUsd": 2.0,
        },
        "failoverMaxUsd": 0.0024,
        "failoverMaxPer1kUsd": 2.4,
    },
    "failover": True,
    "relevance": 1.0,
}

CATALOG_ENTRY = {
    "id": "instagram.profile",
    "slug": "instagram.profile",
    "name": "Instagram Profile",
    "category": "social",
    "description": "Fetch an Instagram account's public profile by handle.",
    "method": "POST",
    "path": "/v1/run/instagram.profile",
    "execution": {"mode": "sync"},
    "provider": "AnyAPI",
    "pricing": SEARCH_RESULT["pricing"],
    "lanes": [
        {
            "pricing": {
                "model": "flat",
                "unit": "request",
                "maxUsd": 0.002,
                "maxPer1kUsd": 2.0,
            },
            "source": {
                "id": "anonymous-giraffe",
                "name": "Giraffe",
                "kind": "anonymous",
                "artworkKey": "giraffe",
            },
        }
    ],
    "tryEligible": True,
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"handle": {"type": "string"}},
        "required": ["handle"],
    },
}


@pytest.fixture
def client():
    fake = MagicMock()
    with patch("getanyapi.AnyAPI", return_value=fake) as constructor:
        fake.constructor = constructor
        yield fake


def test_search_tool_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANYAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"getanyapi\.com/dashboard"):
        AnyApiSearchTool()


def test_describe_tool_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANYAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANYAPI_API_KEY"):
        AnyApiDescribeTool()


def test_run_tool_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANYAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANYAPI_API_KEY"):
        AnyApiRunTool()


def test_key_is_read_from_the_environment(client, monkeypatch):
    monkeypatch.setenv("ANYAPI_API_KEY", "aa_live_from_env")
    AnyApiSearchTool()
    client.constructor.assert_called_once_with(
        api_key="aa_live_from_env", base_url="https://api.getanyapi.com"
    )


def test_search_returns_ranked_catalog_matches(client):
    client.search.return_value = CatalogSearchResults.model_validate(
        {"results": [SEARCH_RESULT], "total": 1, "ranking": "semantic"}
    )

    tool = AnyApiSearchTool(api_key="aa_live_test")
    payload = json.loads(
        tool.run(query="instagram profile", platform="instagram", limit=5)
    )

    client.search.assert_called_once_with(
        query="instagram profile", category=None, platform="instagram", limit=5
    )
    assert payload["total"] == 1
    assert payload["results"][0]["slug"] == "instagram.profile"
    assert payload["results"][0]["pricing"]["from"]["maxUsd"] == 0.002


def test_search_reports_an_api_error(client):
    client.search.side_effect = AnyAPIError("gateway unreachable", status=0)

    tool = AnyApiSearchTool(api_key="aa_live_test")
    result = tool.run(query="instagram profile")

    assert "AnyAPI catalog search failed" in result
    assert "gateway unreachable" in result


def test_describe_returns_the_input_schema(client):
    client.describe.return_value = CatalogEntry.model_validate(CATALOG_ENTRY)

    tool = AnyApiDescribeTool(api_key="aa_live_test")
    payload = json.loads(tool.run(slug="instagram.profile"))

    client.describe.assert_called_once_with("instagram.profile")
    assert payload["inputSchema"]["required"] == ["handle"]
    assert payload["pricing"]["from"]["maxPer1kUsd"] == 2.0


def test_describe_reports_an_api_error(client):
    client.describe.side_effect = AnyAPIError("no such endpoint", status=404)

    tool = AnyApiDescribeTool(api_key="aa_live_test")
    result = tool.run(slug="instagram.nope")

    assert "AnyAPI could not describe 'instagram.nope'" in result
    assert "no such endpoint" in result


def test_run_returns_output_and_usd_cost(client):
    client.run.return_value = RunResult.model_validate(
        {
            "output": {"found": True, "data": {"handle": "nasa", "followers": 100}},
            "provider": "AnyAPI",
            "costUsd": 0.002,
            "items": 1,
            "replayed": False,
        }
    )

    tool = AnyApiRunTool(api_key="aa_live_test")
    payload = json.loads(tool.run(slug="instagram.profile", input={"handle": "nasa"}))

    client.run.assert_called_once_with(
        slug="instagram.profile", input={"handle": "nasa"}
    )
    assert payload["costUsd"] == 0.002
    assert payload["provider"] == "AnyAPI"
    assert payload["output"]["data"]["handle"] == "nasa"


def test_run_reports_an_api_error(client):
    client.run.side_effect = AnyAPIError("upstream refused the input", status=400)

    tool = AnyApiRunTool(api_key="aa_live_test")
    result = tool.run(slug="instagram.profile", input={"handle": "nasa"})

    assert "AnyAPI run failed for 'instagram.profile'" in result
    assert "upstream refused the input" in result


def test_run_reports_the_wallet_balance_when_funds_run_out(client):
    client.run.side_effect = InsufficientBalanceError(
        "insufficient balance", status=402
    )
    client.balance.return_value = Balance(usd=0.01)

    tool = AnyApiRunTool(api_key="aa_live_test")
    result = tool.run(slug="instagram.profile", input={"handle": "nasa"})

    assert "insufficient balance" in result
    assert "$0.01 USD" in result
    assert "https://getanyapi.com/dashboard" in result


def test_run_survives_a_failed_balance_lookup(client):
    client.run.side_effect = InsufficientBalanceError(
        "insufficient balance", status=402
    )
    client.balance.side_effect = AnyAPIError("gateway unreachable", status=0)

    tool = AnyApiRunTool(api_key="aa_live_test")
    result = tool.run(slug="instagram.profile", input={"handle": "nasa"})

    assert result.endswith("insufficient balance")


def test_run_schema_rejects_an_input_that_is_not_an_object():
    with pytest.raises(ValidationError):
        AnyApiRunToolSchema(slug="instagram.profile", input="handle=nasa")


def test_tools_declare_their_optional_dependency(client):
    for tool_class in (AnyApiSearchTool, AnyApiDescribeTool, AnyApiRunTool):
        tool = tool_class(api_key="aa_live_test")
        assert tool.package_dependencies == ["getanyapi"]
        assert [env_var.name for env_var in tool.env_vars] == ["ANYAPI_API_KEY"]
