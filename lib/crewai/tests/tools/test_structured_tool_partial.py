from __future__ import annotations

from functools import partial

from pydantic import BaseModel, Field
import pytest

from crewai.tools.base_tool import Tool
from crewai.tools.structured_tool import CrewStructuredTool


class SearchResult(BaseModel):
    collection: str
    query: str
    limit: int


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run", "_run", "arun", "_arun"])
@pytest.mark.parametrize("nested", [False, True])
async def test_direct_partial_tool_rejects_native_overrides(
    entry: str, nested: bool
) -> None:
    calls: list[tuple[str, str]] = []

    async def lookup(collection: str, query: str) -> str:
        calls.append((collection, query))
        return f"{collection}:{query}"

    def sync_lookup(collection: str, query: str) -> str:
        calls.append((collection, query))
        return f"{collection}:{query}"

    bound = partial(
        lookup if entry in {"arun", "_arun"} else sync_lookup,
        collection="handbook",
        query="travel",
    )
    if nested:
        bound.__dict__["label"] = "inner"
        bound = partial(bound)
    direct = Tool(name="lookup", description="Search the handbook.", func=bound)
    with pytest.raises(
        ValueError, match="Cannot override partial-bound keywords: collection"
    ):
        if entry in {"arun", "_arun"}:
            await getattr(direct, entry)(collection="other")
        else:
            getattr(direct, entry)(collection="other")
    assert calls == []
    assert direct.current_usage_count == 0
    if entry in {"arun", "_arun"}:
        assert await getattr(direct, entry)() == "handbook:travel"
    else:
        assert getattr(direct, entry)() == "handbook:travel"
    assert calls == [("handbook", "travel")]


def search_collection(collection: str, query: str, limit: int = 5) -> SearchResult:
    """Search a collection."""
    return SearchResult(collection=collection, query=query, limit=limit)


def test_direct_partial_tool_infers_unbound_schema() -> None:
    direct = Tool(
        name="lookup",
        description="Search a collection.",
        func=partial(search_collection, collection="handbook"),
    )
    assert set(direct.args_schema.model_fields) == {"query", "limit"}
    assert "handbook" not in direct.formatted_description
    assert direct.run(query="travel", limit="3") == SearchResult(
        collection="handbook",
        query="travel",
        limit=3,
    )


@pytest.mark.parametrize("use_alias", [False, True])
def test_direct_partial_tool_rejects_schema_exposing_bound_keyword(
    use_alias: bool,
) -> None:
    class CollectionArgs(BaseModel):
        collection: str = "other"
        query: str

    class AliasedCollectionArgs(BaseModel):
        index: str = Field(default="other", validation_alias="collection")
        query: str

    with pytest.raises(ValueError, match="args_schema.*collection"):
        Tool(
            name="lookup",
            description="Search the handbook.",
            func=partial(search_collection, collection="handbook"),
            args_schema=AliasedCollectionArgs if use_alias else CollectionArgs,
        )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("fully_bound", [False, True])
async def test_registered_partial_rejects_keyword_override(fully_bound: bool) -> None:
    from crewai import Agent

    def lookup(collection: str, query: str) -> str:
        """Look up a document."""
        return f"{collection}:{query}"

    arguments = {} if fully_bound else {"query": "travel"}
    bound = {"collection": "handbook"}
    if fully_bound:
        bound["query"] = "travel"
    structured = CrewStructuredTool.from_function(partial(lookup, **bound))
    agent = Agent(
        role="Reader",
        goal="Find documents",
        backstory="Test",
        tools=[Tool.from_langchain(structured)],
    )
    registered = agent.tools[0]

    assert registered.run(**arguments) == "handbook:travel"
    with pytest.raises(
        ValueError, match="Cannot override partial-bound keywords: collection"
    ):
        registered.run(collection="other", **arguments)

    async def async_lookup(collection: str, query: str) -> str:
        """Look up a document asynchronously."""
        return lookup(collection, query)

    async_structured = CrewStructuredTool.from_function(partial(async_lookup, **bound))
    async_agent = Agent(
        role="Reader",
        goal="Find documents",
        backstory="Test",
        tools=[Tool.from_langchain(async_structured)],
    )
    async_registered = async_agent.tools[0]
    assert await async_registered.arun(**arguments) == "handbook:travel"
    with pytest.raises(
        ValueError, match="Cannot override partial-bound keywords: collection"
    ):
        await async_registered.arun(collection="other", **arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("fully_bound", [False, True])
async def test_openai_adapter_preserves_registered_partial_bindings(
    fully_bound: bool,
) -> None:
    import json

    pytest.importorskip("agents")
    from crewai import Agent
    from crewai.agents.agent_adapters.openai_agents.openai_agent_tool_adapter import (
        OpenAIAgentToolAdapter,
    )

    def lookup(collection: str, query: str) -> str:
        """Look up a document."""
        return f"{collection}:{query}"

    arguments = {} if fully_bound else {"query": "travel"}
    bound = {"collection": "handbook"}
    if fully_bound:
        bound["query"] = "travel"
    structured = CrewStructuredTool.from_function(partial(lookup, **bound))
    agent = Agent(
        role="Reader",
        goal="Find documents",
        backstory="Test",
        tools=[Tool.from_langchain(structured)],
    )
    adapter = OpenAIAgentToolAdapter()
    adapter.configure_tools(agent.tools)
    openai_tool = adapter.converted_tools[0]

    assert (
        await openai_tool.on_invoke_tool(None, json.dumps(arguments))
        == "handbook:travel"
    )
    with pytest.raises(
        ValueError, match="Cannot override partial-bound keywords: collection"
    ):
        await openai_tool.on_invoke_tool(
            None, json.dumps({"collection": "other", **arguments})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_mode", ["empty", "query_only"])
@pytest.mark.parametrize("use_json", [False, True])
async def test_openai_adapter_rejects_overrides_for_direct_partial_tools(
    schema_mode: str, use_json: bool
) -> None:
    import json

    pytest.importorskip("agents")
    from crewai import Agent
    from crewai.agents.agent_adapters.openai_agents.openai_agent_tool_adapter import (
        OpenAIAgentToolAdapter,
    )

    class QueryArgs(BaseModel):
        query: str

    calls: list[tuple[str, str]] = []

    def lookup(collection: str, query: str) -> str:
        calls.append((collection, query))
        return f"{collection}:{query}"

    if schema_mode == "empty":
        tool = Tool(
            name="lookup",
            description="Look up a document.",
            func=partial(lookup, collection="handbook", query="travel"),
        )
        arguments = {}
    else:
        tool = Tool(
            name="lookup",
            description="Look up a document.",
            func=partial(lookup, collection="handbook"),
            args_schema=QueryArgs,
        )
        arguments = {"query": "travel"}

    agent = Agent(role="Reader", goal="Find documents", backstory="Test", tools=[tool])
    adapter = OpenAIAgentToolAdapter()
    adapter.configure_tools(agent.tools)
    invoke = adapter.converted_tools[0].on_invoke_tool

    malicious = {"collection": "other", **arguments}
    with pytest.raises(
        ValueError, match="Cannot override partial-bound keywords: collection"
    ):
        await invoke(None, json.dumps(malicious) if use_json else malicious)
    assert calls == []
    assert (
        await invoke(None, json.dumps(arguments) if use_json else arguments)
        == "handbook:travel"
    )
    assert calls == [("handbook", "travel")]


@pytest.mark.asyncio
async def test_openai_adapter_checks_zero_argument_payload_before_dispatch() -> None:
    pytest.importorskip("agents")
    from crewai.agents.agent_adapters.openai_agents.openai_agent_tool_adapter import (
        OpenAIAgentToolAdapter,
    )

    calls: list[dict[str, str]] = []

    def lookup(**kwargs: str) -> str:
        calls.append(kwargs)
        return "catalog"

    tool = Tool(name="lookup", description="Find the catalog.", func=lookup)
    adapter = OpenAIAgentToolAdapter([tool])
    adapter.configure_tools([])
    invoke = adapter.converted_tools[0].on_invoke_tool

    with pytest.raises(ValueError, match="Unexpected arguments"):
        await invoke(None, '{"collection": "other"}')
    with pytest.raises(ValueError, match="JSON object"):
        await invoke(None, "[]")
    assert calls == []
    assert await invoke(None, "{}") == "catalog"
    assert calls == [{}]
