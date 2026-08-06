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
respected as fallbacks.

## See also

* [Azure CosmosDB NoSQL vector search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search)
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
