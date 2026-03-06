# OceanBaseVectorSearchTool

## Description

This tool is specifically crafted for conducting semantic searches within documents stored in an OceanBase vector database. Use this tool to find semantically similar documents to a given query using vector similarity search.

OceanBase is a distributed relational database that supports vector storage and similarity search. You can learn more about OceanBase Vector Store here: https://github.com/oceanbase/pyobvector

## Installation

Install the crewai_tools package with OceanBase support by executing the following command in your terminal:

```shell
pip install 'crewai-tools[pyobvector]'
```

or

```shell
uv pip install 'crewai[tools] crewai-tools[pyobvector]'
```

If you use the default OpenAI embeddings, also ensure `openai` is installed:

```shell
uv pip install 'crewai[tools] crewai-tools[pyobvector] openai'
```

## Prerequisites

- OceanBase database version >= 4.3.3.0 (with vector store support)
- A table with a vector column and vector index created (or create it as in **Setting up the table** below)
- Connection credentials: URI (host:port), user, password, database name

## Example

To utilize the OceanBaseVectorSearchTool for different use cases, follow these examples:

```python
from crewai_tools import OceanBaseVectorSearchTool, OceanBaseConfig
from crewai import Agent

# Configure OceanBase connection (OceanBaseConfig is from crewai_tools)
config = OceanBaseConfig(
    uri="127.0.0.1:2881",
    user="root",
    password="",
    db_name="crewai",
    table_name="documents",
    vec_column_name="embedding",
    limit=5,
    distance_func="l2_distance",
)

tool = OceanBaseVectorSearchTool(oceanbase_config=config)

# Adding the tool to an agent
rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions with the help of the OceanBaseVectorSearchTool. Retrieve the most relevant documents from the OceanBase database.",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

With custom embedding (no OpenAI key required):

```python
from crewai_tools import OceanBaseVectorSearchTool, OceanBaseConfig

def my_embedding(text: str) -> list[float]:
    # Return a fixed-dimension vector, e.g. from your own model
    return [0.1] * 1536

config = OceanBaseConfig(
    uri="127.0.0.1:2881",
    user="root",
    password="",
    db_name="crewai",
    table_name="documents",
    limit=5,
    distance_threshold=0.5,
    output_columns=["id", "title", "content"],
)

tool = OceanBaseVectorSearchTool(
    oceanbase_config=config,
    custom_embedding_fn=my_embedding,
)
```

Filtering by column when calling the tool:

```python
result = tool._run(
    query="What is machine learning?",
    filter_by="category",
    filter_value="AI",
)
```

## Arguments

### OceanBaseConfig (passed as `oceanbase_config`)

| Argument | Required | Description |
|----------|----------|-------------|
| `uri` | Yes | OceanBase connection URI (e.g. `"127.0.0.1:2881"`). |
| `user` | Yes | OceanBase user (e.g. `"root"` or `"root@test"` for tenant). |
| `password` | No | OceanBase password (default: `""`). |
| `db_name` | Yes | OceanBase database name. |
| `table_name` | Yes | Table name to search in. |
| `vec_column_name` | No | Vector column name in the table (default: `"embedding"`). |
| `limit` | No | Number of results to return (default: `3`). |
| `distance_threshold` | No | Only return results with distance <= this value (default: `None`). |
| `distance_func` | No | One of `"l2_distance"`, `"cosine_distance"`, `"inner_product"`, `"negative_inner_product"` (default: `"l2_distance"`). |
| `output_columns` | No | List of column names to return (default: `None` = all columns). |

### OceanBaseVectorSearchTool

| Argument | Required | Description |
|----------|----------|-------------|
| `oceanbase_config` | Yes | An `OceanBaseConfig` instance. |
| `custom_embedding_fn` | No | Callable `(str) -> list[float]`. If not set, uses OpenAI embeddings (requires `OPENAI_API_KEY`). |
| `pyobvector_package` | No | Package path for pyobvector (default: `"pyobvector"`). |

### Tool schema (agent usage)

- `query` (required): Query text to search for.
- `filter_by` (optional): Column name to filter by.
- `filter_value` (optional): Value to filter by (use together with `filter_by`).

## Setting up the table

Before using the tool, create a table with a vector column and vector index in OceanBase. Example using pyobvector:

```python
from pyobvector import ObVecClient, VECTOR
from sqlalchemy import Column, Integer, VARCHAR, Text

client = ObVecClient(
    uri="127.0.0.1:2881",
    user="root",
    password="",
    db_name="crewai",
)

table_name = "documents"
cols = [
    Column("id", Integer, primary_key=True),
    Column("title", VARCHAR(255)),
    Column("content", Text),
    Column("embedding", VECTOR(1536)),  # Match your embedding model dimension
]

client.create_table(table_name, columns=cols)
client.create_index(
    table_name,
    is_vec_index=True,
    index_name="vidx_embedding",
    column_names=["embedding"],
    vidx_params="distance=l2, type=hnsw, lib=vsag",
)

# Insert rows with id, title, content, embedding (list of floats)
# client.insert(table_name, data=[...])
```

## Notes

- Default embeddings use OpenAI `text-embedding-3-large`; set `OPENAI_API_KEY` or pass `custom_embedding_fn`.
- OceanBase version must be >= 4.3.3.0 for vector support.
- Use the same `distance_func` as when creating the vector index.
- Results include a distance score; lower distance means higher similarity.
