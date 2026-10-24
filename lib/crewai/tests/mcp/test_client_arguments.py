from crewai.mcp.client import MCPClient
from crewai.mcp.transports.stdio import StdioTransport


def make_client() -> MCPClient:
    return MCPClient(StdioTransport(command="python", args=["server.py"]))


def test_clean_tool_arguments_preserves_explicit_empty_containers() -> None:
    client = make_client()

    cleaned = client._clean_tool_arguments(
        {
            "empty_dict": {},
            "empty_list": [],
            "nested": {"removed": None},
            "items": [{"removed": None}, 0, "", None],
        }
    )

    assert cleaned == {
        "empty_dict": {},
        "empty_list": [],
        "nested": {},
        "items": [{}, 0, ""],
    }


def test_clean_tool_arguments_removes_none_values_recursively() -> None:
    client = make_client()

    cleaned = client._clean_tool_arguments(
        {
            "removed": None,
            "nested": {"kept": "value", "removed": None},
            "items": [{"kept": True, "removed": None}, None],
        }
    )

    assert cleaned == {
        "nested": {"kept": "value"},
        "items": [{"kept": True}],
    }


def test_clean_tool_arguments_preserves_empty_sources_list() -> None:
    client = make_client()

    assert client._clean_tool_arguments({"sources": []}) == {"sources": []}
