# Azure CosmosDB NoSQL Tools

Three CrewAI `BaseTool` integrations backed by **Azure CosmosDB for NoSQL**:

| Tool | Purpose |
| ---- | ------- |
| `AzureCosmosDBNoSqlSearchTool` | Vector / full-text / hybrid search over a Cosmos NoSQL container |
| `AzureCosmosDBSemanticCacheTool` | Semantic cache for LLM responses with TTL support |
| `AzureCosmosDBMemoryTool` | CRUD over agent memory items with hierarchical partition keys |

## Installation

These tools live behind an optional extra so the base `crewai-tools` install
stays small:

```bash
pip install 'crewai-tools[azure-cosmosdb]'
```

This pulls in `azure-cosmos`, `azure-identity`, and `openai`.

## Authentication

All three tools accept either an account key **or** an
`azure.core.credentials.TokenCredential` (e.g. `DefaultAzureCredential`):

```python
from azure.identity import DefaultAzureCredential

tool = AzureCosmosDBNoSqlSearchTool(
    cosmos_host="https://<account>.documents.azure.com:443/",
    token_credential=DefaultAzureCredential(),
    indexing_policy=...,
    cosmos_container_properties={"partition_key": {"paths": ["/agent_id"], "kind": "Hash"}},
)
```

For embeddings, set either `azure_openai_endpoint` (Azure OpenAI) or
`openai_api_key` (standard OpenAI). Environment variables
`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `OPENAI_API_KEY` are
respected as fallbacks. You can also pass a custom `embedder` exposing
`embed_documents` / `embed_query` to bypass OpenAI entirely.

## Usage

### `AzureCosmosDBNoSqlSearchTool` — vector / full-text / hybrid search

```python
from crewai import Agent
from crewai_tools import AzureCosmosDBNoSqlSearchTool, AzureCosmosDBNoSqlSearchConfig

indexing_policy = {
    "indexingMode": "consistent",
    "automatic": True,
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
}
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "dimensions": 1536,
            "distanceFunction": "cosine",
        }
    ]
}

search_tool = AzureCosmosDBNoSqlSearchTool(
    cosmos_host="https://<account>.documents.azure.com:443/",
    key="<account-key>",
    database_name="crewAI_database",
    container_name="crewAI_container",
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    cosmos_container_properties={"partition_key": {"paths": ["/id"], "kind": "Hash"}},
    embedding_model="text-embedding-3-large",
    dimensions=1536,
    search_type="vector",  # or vector_score_threshold / full_text_search /
                           # full_text_ranking / hybrid / hybrid_score_threshold
    query_config=AzureCosmosDBNoSqlSearchConfig(max_results=5),
)

# Ingest documents (embeds text and writes them in batches):
search_tool.add_texts(["CrewAI orchestrates role-playing autonomous agents."])

agent = Agent(
    role="Research Assistant",
    goal="Find relevant information in the knowledge base",
    tools=[search_tool],
)
```

Full-text and hybrid searches require `full_text_search_enabled=True`, a
`fullTextIndexes` entry in the indexing policy, and a `full_text_policy`.

### `AzureCosmosDBSemanticCacheTool` — semantic cache for LLM responses

```python
from crewai_tools import (
    AzureCosmosDBSemanticCacheTool,
    AzureCosmosDBSemanticCacheConfig,
)

config = AzureCosmosDBSemanticCacheConfig(
    cosmos_host="https://<account>.documents.azure.com:443/",
    key="<account-key>",
    indexing_policy=indexing_policy,
    vector_embedding_policy=vector_embedding_policy,
    full_text_policy={"fullTextPaths": [{"path": "/prompt", "language": "en-US"}]},
    similarity_threshold=0.85,
    default_ttl=86400,          # seconds; None disables expiry
    enable_hybrid_search=True,  # set False for vector-only (no full-text policy)
    llm_string="gpt-4o-2024-08-06",
)
cache_tool = AzureCosmosDBSemanticCacheTool(config=config)

# operation is one of: 'search', 'update', 'clear'
cache_tool.run(operation="update", prompt="What is CrewAI?", response="A multi-agent framework.")
cache_tool.run(operation="search", prompt="What's CrewAI?")   # -> cache hit
cache_tool.run(operation="clear")                              # clears this llm_string namespace
```

### `AzureCosmosDBMemoryTool` — agent memory CRUD

```python
from crewai_tools import AzureCosmosDBMemoryTool, AzureCosmosDBMemoryConfig

config = AzureCosmosDBMemoryConfig(
    cosmos_host="https://<account>.documents.azure.com:443/",
    key="<account-key>",
    database_name="memory_database",
    container_name="memory_container",
    cosmos_container_properties={
        "partition_key": {"paths": ["/agent_id"], "kind": "Hash"}
    },
    use_optimistic_concurrency=False,  # True -> guard updates with the doc _etag
)
memory_tool = AzureCosmosDBMemoryTool(config=config)

# operation is one of: 'store', 'read', 'retrieve', 'update', 'delete', 'clear'
memory_tool.run(
    operation="store",
    memory_item={"id": "m-1", "agent_id": "a-1", "content": {"text": "user likes dark mode"}},
)
memory_tool.run(operation="read", partition_key_value="a-1", memory_id="m-1")
memory_tool.run(operation="retrieve", partition_key_value="a-1", max_results=10)
memory_tool.run(operation="delete", partition_key_value="a-1", memory_id="m-1")
```

Both configs also accept `connection_string` (instead of `cosmos_host` + `key`)
and `token_credential` for AAD auth. Hierarchical partition keys are supported
by passing multiple `paths` and list-valued `partition_key_value`.

## Testing

Unit tests are fully mocked (no live account or extras required) and run in CI.

A separate **live integration suite**
(`tests/tools/azure/cosmosdb_nosql/test_integration_cosmosdb.py`) exercises the
real service end-to-end. It is **skipped unless `AZURE_COSMOS_URI` is set**, so
CI stays hermetic. It uses a deterministic embedder, so no OpenAI key is needed.

Authenticate with an account key **or** AAD/RBAC — set `AZURE_COSMOS_KEY` for
key auth, or leave it unset to use `DefaultAzureCredential` (managed identity,
`az login`, service-principal env vars, ...):

```bash
export AZURE_COSMOS_URI="https://<account>.documents.azure.com:443/"

# Key auth:
export AZURE_COSMOS_KEY="<account-key>"
# ...or AAD/RBAC: leave AZURE_COSMOS_KEY unset and run `az login` first.

uv run pytest lib/crewai-tools/tests/tools/azure/cosmosdb_nosql/test_integration_cosmosdb.py
```

## See also

* [Azure CosmosDB NoSQL vector search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search)
* [Full-text search](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search)
* [Hybrid search](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search)
* [Hierarchical partition keys](https://learn.microsoft.com/azure/cosmos-db/hierarchical-partition-keys)
* [Container TTL](https://learn.microsoft.com/azure/cosmos-db/nosql/time-to-live)

For a native CrewAI **memory backend** (vs. an agent-callable tool), see
`crewai.memory.storage.cosmosdb_nosql_storage.CosmosDBNoSqlStorage`.

It can be selected directly from `Memory`:

```python
from crewai.memory.unified_memory import Memory

# Reads AZURE_COSMOS_CONNECTION_STRING, or AZURE_COSMOS_HOST (+ AZURE_COSMOS_KEY,
# else DefaultAzureCredential). Optional: AZURE_COSMOS_DATABASE_NAME,
# AZURE_COSMOS_CONTAINER_NAME, AZURE_COSMOS_VECTOR_DIM.
memory = Memory(storage="cosmosdb")

# ...or pass a pre-configured instance for full control:
from crewai.memory.storage.cosmosdb_nosql_storage import CosmosDBNoSqlStorage

memory = Memory(storage=CosmosDBNoSqlStorage.from_env())
```
