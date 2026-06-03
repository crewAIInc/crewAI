# OracleVectorSearchTool

## Description

This tool is specifically crafted for conducting vector searches within Oracle AI Vector Search tables. Use this tool to find semantically similar documents stored in Oracle Database.

Oracle Database 23ai and later can store vectors natively and query them with `vector_distance(...)`. This tool follows CrewAI's existing vector-search tool shape while using Oracle-native SQL under the hood.

## Installation

Install the CrewAI tools package with Oracle support:

```shell
pip install crewai-tools[oracledb]
```

or

```shell
uv add crewai-tools --extra oracledb
```

## Example

```python
from crewai_tools import (
    OracleVectorSearchConfig,
    OracleVectorSearchQueryConfig,
    OracleVectorSearchTool,
)

tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(
        user=os.environ["ORACLE_DB_USERNAME"],
        password=os.environ["ORACLE_DB_SECRET"],
        dsn=os.environ["ORACLE_DB_DSN"],
        table_name="DOCS_VECTORS",
        limit=3,
        distance_strategy="COSINE",
    ),
    query_config=OracleVectorSearchQueryConfig(
        score_threshold=0.6,
        filter={"source": "docs"},
    ),
)
```

Running a search with native JSON numeric filtering:

```python
results = tool._run(
    query="oracle vector",
    filter_by="priority",
    filter_value=5,
    score_threshold=0.6,
)
```

Using richer Oracle-style filters:

```python
results = tool._run(
    query="oracle vector",
    filters='{"$or":[{"source":"docs"},{"priority":{"$gte":3}}]}',
)
```

Using a custom embedding function:

```python
tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(
        user=os.environ["ORACLE_DB_USERNAME"],
        password=os.environ["ORACLE_DB_SECRET"],
        dsn=os.environ["ORACLE_DB_DSN"],
        table_name="DOCS_VECTORS",
    ),
    embedding_function=my_embedding_function,
)
```

Passing additional `oracledb.connect()` options:

```python
tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(
        user=os.environ["ORACLE_DB_USERNAME"],
        password=os.environ["ORACLE_DB_SECRET"],
        dsn=os.environ["ORACLE_DB_DSN"],
        table_name="DOCS_VECTORS",
        connection_kwargs={
            "config_dir": "/path/to/wallet",
            "wallet_location": "/path/to/wallet",
        },
    ),
    embedding_function=my_embedding_function,
)
```

Using a caller-managed connection pool:

```python
pool = oracledb.create_pool(
    user=os.environ["ORACLE_DB_USERNAME"],
    password=os.environ["ORACLE_DB_SECRET"],
    dsn=os.environ["ORACLE_DB_DSN"],
    min=1,
    max=4,
)

tool = OracleVectorSearchTool(
    oracle_config=OracleVectorSearchConfig(table_name="DOCS_VECTORS"),
    client=pool,
    embedding_function=my_embedding_function,
)
```

Preloading data into Oracle:

```python
tool.create_table()
tool.add_texts(
    ["CrewAI integrates with Oracle AI Vector Search."],
    metadatas=[{"source": "docs"}],
)
tool.create_vector_index(
    idx_type="HNSW",
    params={"accuracy": 90, "neighbors": 32, "efconstruction": 200, "parallel": 8},
)

# Or create an IVF index instead.
tool.create_vector_index(
    index_name="DOCS_IVF_IDX",
    idx_type="IVF",
    params={
        "accuracy": 90,
        "neighbor_partitions": 32,
        "samples_per_partition": 1,
        "min_vectors_per_partition": 0,
        "parallel": 8,
    },
)
```

## Arguments

- `oracle_config`: Oracle connection and search settings. Required.
- `query_config`: Optional default query behavior including `limit`, `score_threshold`, and Oracle-style metadata filters.
- `embedding_function`: Optional callable used instead of OpenAI embeddings.
- `embedding_model`: OpenAI embedding model used when `embedding_function` is not provided.
- `dimensions`: Embedding dimension used when creating tables and validating inserted embeddings.

`oracle_config` supports:

- `user`, `password`, `dsn`: Common Oracle connection fields when a client is not provided.
- `connection_kwargs`: Optional extra keyword arguments passed to `oracledb.connect()` when the tool creates the connection.
- `table_name`: Oracle table containing your text, metadata, and vector columns.
- `limit`: Number of search results to return.
- `score_threshold`: Optional maximum vector distance. Only rows with `distance <= score_threshold` are returned.
- `distance_strategy`: One of `COSINE`, `EUCLIDEAN`, or `DOT`.
- `index_name`: Optional default vector index name used by `create_vector_index()`.

`create_vector_index()` supports:

- `idx_type`: `HNSW` or `IVF`. Defaults to `HNSW`.
- `params`: Oracle vector index parameters. For `HNSW`, use `accuracy`, `neighbors`, `efconstruction`, and `parallel`. For `IVF`, use `accuracy`, `neighbor_partitions`, `samples_per_partition`, `min_vectors_per_partition`, and `parallel`.

`client` may be a caller-managed `oracledb.Connection` or `oracledb.ConnectionPool`. Pools must be created by the caller and passed through `client`; `OracleVectorSearchConfig` only configures single connections created with `oracledb.connect()`.

The tool creates and expects a fixed Oracle schema:
- `id`
- `text`
- `metadata` as native Oracle `JSON`
- `embedding`

`_run()` also supports:

- `filters`: JSON string for richer Oracle metadata filters such as `{"$or":[{"source":"docs"},{"topic":{"$eq":"oracle"}}]}`
- `limit`: Per-call result limit override
- `score_threshold`: Per-call maximum distance override

Result format:

- `context`: The matched text payload.
- `metadata`: Oracle JSON metadata decoded back into Python values.
- `distance`: Raw Oracle `vector_distance(...)` value.
- `score`: Kept for consistency with other CrewAI vector tools, but currently equal to `distance`.
