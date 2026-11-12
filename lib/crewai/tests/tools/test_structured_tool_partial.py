from __future__ import annotations

from functools import partial

from pydantic import BaseModel, Field
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


def test_partial_tool_unwraps_nested_partials() -> None:
    first = partial(search_collection, "handbook")
    # A partial with attributes retains its own identity when wrapped again.
    first.__dict__["label"] = "handbook"
    nested = partial(first, limit=2)

    tool = CrewStructuredTool.from_function(nested)

    assert tool.name == "search_collection"
    assert tool.result_schema is SearchResult
    assert set(tool.args) == {"query"}
    assert tool.invoke({"query": "travel"}) == SearchResult(
        collection="handbook", query="travel", limit=2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True])
async def test_partial_tool_keeps_keyword_bindings_out_of_schema(
    async_path: bool,
) -> None:
    tool = CrewStructuredTool.from_function(
        partial(search_collection, collection="handbook")
    )

    assert set(tool.args) == {"query", "limit"}
    assert "handbook" not in tool.formatted_description
    if async_path:
        result = await tool.ainvoke({"query": "travel"})
    else:
        result = tool.invoke({"query": "travel"})
    assert result == SearchResult(collection="handbook", query="travel", limit=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_path", [False, True])
@pytest.mark.parametrize(
    ("payload", "call_kwargs"),
    [
        ({"query": "travel", "collection": "other"}, {}),
        ('{"query":"travel","collection":"other"}', {}),
        ({"query": "travel"}, {"collection": "other"}),
    ],
)
async def test_partial_tool_rejects_overrides_of_bound_keywords(
    async_path: bool,
    payload: str | dict[str, str],
    call_kwargs: dict[str, str],
) -> None:
    tool = CrewStructuredTool.from_function(
        partial(search_collection, collection="handbook")
    )

    with pytest.raises(
        ValueError, match="Cannot override partial-bound keyword.*collection"
    ):
        if async_path:
            await tool.ainvoke(payload, **call_kwargs)
        else:
            tool.invoke(payload, **call_kwargs)
    assert tool.current_usage_count == 0


@pytest.mark.parametrize("use_alias", [False, True])
def test_partial_tool_rejects_bound_keyword_in_explicit_schema(
    use_alias: bool,
) -> None:
    class ExposedCollection(BaseModel):
        collection: str = "handbook"
        query: str

    class ExposedCollectionAlias(BaseModel):
        index: str = Field(default="handbook", validation_alias="collection")
        query: str

    with pytest.raises(ValueError, match="args_schema.*collection"):
        CrewStructuredTool.from_function(
            partial(search_collection, collection="handbook"),
            args_schema=ExposedCollectionAlias if use_alias else ExposedCollection,
        )
