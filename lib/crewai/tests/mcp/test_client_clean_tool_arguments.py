from crewai.mcp.client import MCPClient
from crewai.mcp.transports.stdio import StdioTransport


def _client() -> MCPClient:
    return MCPClient(StdioTransport(command="python", args=["server.py"]))


def test_explicit_empty_containers_reach_the_server():
    cleaned = _client()._clean_tool_arguments(
        {"query": "cats", "tags": [], "filters": {}, "sources": []}
    )
    assert cleaned == {
        "query": "cats",
        "tags": [],
        "filters": {},
        "sources": [],
    }


def test_list_items_that_clean_to_empty_keep_their_position():
    cleaned = _client()._clean_tool_arguments({"mixed": [{"opt": None}, 1]})
    assert cleaned == {"mixed": [{}, 1]}


def test_none_values_are_still_removed():
    cleaned = _client()._clean_tool_arguments(
        {
            "query": "cats",
            "limit": None,
            "nested": {"keep": 1, "drop": None},
            "items": [None, {"a": None, "b": 2}],
        }
    )
    assert cleaned == {
        "query": "cats",
        "nested": {"keep": 1},
        "items": [{"b": 2}],
    }


def test_falsy_scalars_are_preserved():
    cleaned = _client()._clean_tool_arguments({"label": "", "count": 0, "flag": False})
    assert cleaned == {"label": "", "count": 0, "flag": False}
