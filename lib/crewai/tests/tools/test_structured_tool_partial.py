from __future__ import annotations

from functools import partial

from pydantic import BaseModel
import pytest

from crewai.tools.structured_tool import CrewStructuredTool


class SearchResult(BaseModel):
    collection: str
    query: str
    limit: int


def search_collection(collection: str, query: str, limit: int = 5) -> SearchResult:
    """Search a collection."""
    return SearchResult(collection=collection, query=query, limit=limit)


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_partial_tool_infers_only_unbound_arguments(explicit_metadata: bool) -> None:
    bound_search = partial(search_collection, "handbook")
    if explicit_metadata:
        tool = CrewStructuredTool.from_function(
            bound_search, name="search_handbook", description="Search the handbook."
        )
        assert tool.name == "search_handbook"
        assert tool.description == "Search the handbook."
    else:
        tool = CrewStructuredTool.from_function(bound_search)
        assert tool.name == "search_collection"
        assert tool.description == "Search a collection."

    assert tool.args_schema is not None
    assert set(tool.args_schema.model_fields) == {"query", "limit"}
    assert tool.result_schema is SearchResult
    assert tool.invoke({"query": "travel", "limit": "3"}) == SearchResult(
        collection="handbook", query="travel", limit=3
    )
