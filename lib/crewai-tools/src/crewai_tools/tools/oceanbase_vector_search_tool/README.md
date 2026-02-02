# OceanBaseVectorSearchTool

## Description

This tool is designed for performing vector similarity searches within an OceanBase database. OceanBase is a distributed relational database developed by Ant Group that supports native vector indexing and search capabilities using HNSW (Hierarchical Navigable Small World) algorithm.

Use this tool to find semantically similar documents to a given query by leveraging OceanBase's vector search functionality.

For more information about OceanBase vector capabilities, see:
https://en.oceanbase.com/docs/common-oceanbase-database-10000000001976351

## Installation

Install the crewai_tools package with OceanBase support by executing the following command in your terminal:

```shell
pip install crewai-tools[oceanbase]
```

or

```shell
uv add crewai-tools --extra oceanbase
```

## Example

### Basic Usage

```python
from crewai_tools import OceanBaseVectorSearchTool

tool = OceanBaseVectorSearchTool(
    connection_uri="127.0.0.1:2881",
    user="root@test",
    password="",
    db_name="test",
    table_name="documents",
)
```

### With Custom Configuration

```python
from crewai_tools import OceanBaseVectorSearchConfig, OceanBaseVectorSearchTool

query_config = OceanBaseVectorSearchConfig(
    limit=10,
    distance_func="cosine",
    distance_threshold=0.5,
)

tool = OceanBaseVectorSearchTool(
    connection_uri="127.0.0.1:2881",
    user="root@test",
    password="your_password",
    db_name="my_database",
    table_name="my_documents",
    vector_column_name="embedding",
    text_column_name="content",
    metadata_column_name="metadata",
    query_config=query_config,
    embedding_model="text-embedding-3-large",
    dimensions=3072,
)
```

### Adding the Tool to an Agent

```python
from crewai import Agent
from crewai_tools import OceanBaseVectorSearchTool

tool = OceanBaseVectorSearchTool(
    connection_uri="127.0.0.1:2881",
    user="root@test",
    db_name="test",
    table_name="documents",
)

rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions using the OceanBaseVectorSearchTool.",
    goal="Answer user questions by searching relevant documents",
    backstory="You have access to a knowledge base stored in OceanBase",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

### Preloading Documents

```python
from crewai_tools import OceanBaseVectorSearchTool
import os

tool = OceanBaseVectorSearchTool(
    connection_uri="127.0.0.1:2881",
    user="root@test",
    db_name="test",
    table_name="documents",
)

texts = []
metadatas = []
for filename in os.listdir("knowledge"):
    with open(os.path.join("knowledge", filename), "r") as f:
        texts.append(f.read())
        metadatas.append({"source": filename})

tool.add_texts(texts, metadatas=metadatas)
```

## Configuration Options

### OceanBaseVectorSearchConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 4 | Number of documents to return |
| `distance_func` | str | "l2" | Distance function: "l2", "cosine", or "inner_product" |
| `distance_threshold` | float | None | Only return results with distance <= threshold |
| `include_embeddings` | bool | False | Whether to include embedding vectors in results |

### OceanBaseVectorSearchTool

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `connection_uri` | str | Yes | OceanBase connection URI (e.g., "127.0.0.1:2881") |
| `user` | str | Yes | Username for connection (e.g., "root@test") |
| `password` | str | No | Password for connection |
| `db_name` | str | No | Database name (default: "test") |
| `table_name` | str | Yes | Table containing vector data |
| `vector_column_name` | str | No | Column with embeddings (default: "embedding") |
| `text_column_name` | str | No | Column with text content (default: "text") |
| `metadata_column_name` | str | No | Column with metadata (default: "metadata") |
| `embedding_model` | str | No | OpenAI model for embeddings (default: "text-embedding-3-large") |
| `dimensions` | int | No | Embedding dimensions (default: 1536) |
| `query_config` | OceanBaseVectorSearchConfig | No | Search configuration |

## Environment Variables

- `OPENAI_API_KEY`: Required for generating embeddings
- `AZURE_OPENAI_ENDPOINT`: Optional, for Azure OpenAI support
