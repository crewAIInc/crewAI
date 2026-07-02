# DB2 Vector Search Tool

IBM DB2 Vector Search Tool for CrewAI.

Supports:

- IBM DB2 native VECTOR search
- OpenAI embeddings
- Custom embedding functions
- Metadata filtering
- Runtime dynamic imports
- Standardized CrewAI tool architecture

---

# Installation

```bash
pip install ibm_db ibm_db_dbi openai
```

---

# Environment Variables

```env
OPENAI_API_KEY=your_openai_key

DB2_DATABASE=TESTDB
DB2_HOSTNAME=localhost
DB2_USERNAME=db2user
DB2_PASSWORD=password
```

---

# Example Usage

```python
from db2_search_tool import (
    DB2Config,
    DB2VectorSearchTool,
)

config = DB2Config(
    database="TESTDB",
    hostname="localhost",
    table_name="documents",
)

tool = DB2VectorSearchTool(
    db2_config=config
)

result = tool._run(
    query="What is machine learning?",
)

print(result)
```

---

# Example With Metadata Filtering

```python
result = tool._run(
    query="AI papers",
    filter_by="category",
    filter_value="AI",
)
```

---

# Supported Features

- DB2 VECTOR datatype
- VECTOR_DISTANCE search
- COSINE similarity
- Metadata filtering
- OpenAI embedding fallback
- Custom embedding functions

---

# Architecture

This tool follows the same architecture as:

- QdrantVectorSearchTool
- WeaviateVectorSearchTool

Responsibilities:

- Generate query embeddings
- Perform vector similarity search
- Apply optional metadata filters
- Return normalized JSON results

This tool is retrieval-only.

Document ingestion should be handled separately.