# SochDBVectorSearchTool

## Description

The `SochDBVectorSearchTool` lets CrewAI agents search a SochDB collection over gRPC using vector similarity. Use it when an agent needs grounded context from a SochDB-backed knowledge base, memory store, or retrieval system.

SochDB supports embedded and hosted deployments. This CrewAI tool targets the gRPC client path so agents can connect to a local or remote SochDB server.

## Installation

Install CrewAI tools with the SochDB extra:

```shell
uv add "crewai-tools[sochdb]" openai
```

Or install the required packages manually:

```shell
pip install crewai-tools sochdb openai
```

## Example

```python
from crewai import Agent
from crewai_tools import SochDBConfig, SochDBVectorSearchTool

tool = SochDBVectorSearchTool(
    sochdb_config=SochDBConfig(
        grpc_address="studio.agentslab.host:50053",
        collection_name="knowledge",
        namespace="default",
        limit=3,
    )
)

rag_agent = Agent(
    role="Knowledge retrieval specialist",
    goal="Answer questions using relevant SochDB context.",
    backstory="You retrieve grounded context from SochDB before answering.",
    tools=[tool],
)
```

By default the tool creates query embeddings with OpenAI, so set `OPENAI_API_KEY` in the environment. You can provide `custom_embedding_fn` to use an embedding provider that matches the vectors stored in your SochDB collection.

```python
tool = SochDBVectorSearchTool(
    sochdb_config=SochDBConfig(
        grpc_address="localhost:50051",
        collection_name="support_docs",
    ),
    custom_embedding_fn=lambda query: my_embedder.embed_query(query),
)
```

## Arguments

- `grpc_address`: SochDB gRPC endpoint, for example `localhost:50051` or `studio.agentslab.host:50053`.
- `collection_name`: Collection to search.
- `namespace`: SochDB namespace. Defaults to `default`.
- `limit`: Number of matches to return. Defaults to `3`.
- `metadata_filter_json`: Optional tool input containing a JSON object used to filter document metadata.
- `custom_embedding_fn`: Optional callable or import path that receives a query string and returns one embedding vector.
- `embedding_model`: OpenAI embedding model used when `custom_embedding_fn` is not provided. Defaults to `text-embedding-3-large`.
