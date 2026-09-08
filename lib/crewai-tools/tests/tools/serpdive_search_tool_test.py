import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from serpdive import SearchResponse

from crewai_tools.tools.serpdive_search_tool.serpdive_search_tool import (
    SerpdiveSearchTool,
)

PAYLOAD = {
    "query": "seine temperature",
    "results": [
        {
            "url": "https://a.example",
            "title": "T",
            "date": "2026-07-01",
            "content": "C1",
        },
        {"url": "https://b.example", "title": "U", "content": "C2"},
    ],
}


def make_tool(**kwargs):
    tool = SerpdiveSearchTool(api_key="sd_live_TEST", **kwargs)
    tool.client = MagicMock()
    tool.client.search.return_value = SearchResponse.from_dict(PAYLOAD)
    tool.async_client = MagicMock()
    tool.async_client.search = AsyncMock(
        return_value=SearchResponse.from_dict(PAYLOAD)
    )
    return tool


def test_tool_initialization_defaults():
    tool = SerpdiveSearchTool(api_key="sd_live_TEST")
    assert tool.name == "SERPdive Search"
    assert tool.model == "mako"
    assert tool.answer is False
    assert tool.max_results is None


def test_run_returns_json_payload():
    tool = make_tool()
    out = json.loads(tool._run(query="seine temperature"))
    assert out["query"] == "seine temperature"
    assert out["results"][0] == {
        "url": "https://a.example",
        "title": "T",
        "date": "2026-07-01",
        "content": "C1",
    }
    # no date key when the API returned none
    assert "date" not in out["results"][1]
    # answer key absent when not requested
    assert "answer" not in out


def test_params_travel_to_the_sdk():
    tool = make_tool(model="moby", answer=True, max_results=3)
    tool._run(query="q")
    tool.client.search.assert_called_once_with(
        "q", model="moby", answer=True, max_results=3
    )


@pytest.mark.asyncio
async def test_arun_returns_json_payload():
    tool = make_tool()
    out = json.loads(await tool._arun(query="seine temperature"))
    assert [r["url"] for r in out["results"]] == [
        "https://a.example",
        "https://b.example",
    ]
