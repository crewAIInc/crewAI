# QdrantVectorSearchTool

## Description

This tool is specifically crafted for conducting semantic searches within docs within a Qdrant vector database. Use this tool to find semantically similar docs to a given query.

Qdrant is a vector database that is used to store and query vector embeddings. You can follow their docs here: https://qdrant.tech/documentation/

## Installation

Install the crewai_tools package by executing the following command in your terminal:

```shell
uv pip install 'crewai[tools] qdrant-client openai'
```

## Example

To utilize the QdrantVectorSearchTool for different use cases, follow these examples: Default model is openai.

```python
from crewai_tools import QdrantVectorSearchTool

# To enable the tool to search any website the agent comes across or learns about during its operation
tool = QdrantVectorSearchTool(
    collection_name="example_collections",
    limit=3,
    qdrant_url="https://your-qdrant-cluster-url.com",
    qdrant_api_key="your-qdrant-api-key", # (optional)
)


# Adding the tool to an agent
rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions with the help of the QdrantVectorSearchTool. Retrieve the most relevant docs from the Qdrant database.",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

## Arguments

- `collection_name` : The name of the collection to search within. (Required)
- `qdrant_url` : The URL of the Qdrant cluster. (Required)
- `qdrant_api_key` : The API key for the Qdrant cluster. (Optional)
- `limit` : The number of results to return. (Optional)
- `vectorizer` : The vectorizer to use. (Optional)

