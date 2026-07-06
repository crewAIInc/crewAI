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

DB2_CONNECTION_STRING=DATABASE=TESTDB;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=db2user;PWD=password;
```

---

# Example Usage

```python
from crewai_tools.tools.db2_search_tool import DB2VectorSearchTool

tool = DB2VectorSearchTool(
    connection_string="DATABASE=TESTDB;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=db2user;PWD=password;",
    table_name="documents",
)

result = tool.run(
    query="What is machine learning?",
)

print(result)
```

---

# Example With Metadata Filtering

```python
result = tool.run(
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
- Uses a custom embedding function if supplied, otherwise OpenAI embeddings

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