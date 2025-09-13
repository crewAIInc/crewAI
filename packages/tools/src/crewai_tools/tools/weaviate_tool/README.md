# WeaviateVectorSearchTool

## Description
This tool is specifically crafted for conducting semantic searches within docs within a Weaviate vector database. Use this tool to find semantically similar docs to a given query.

Weaviate is a vector database that is used to store and query vector embeddings. You can follow their docs here: https://weaviate.io/developers/wcs/connect

## Installation
Install the crewai_tools package by executing the following command in your terminal:

```shell
uv pip install 'crewai[tools]'
```

## Example
To utilize the WeaviateVectorSearchTool for different use cases, follow these examples:

```python
from crewai_tools import WeaviateVectorSearchTool

# To enable the tool to search any website the agent comes across or learns about during its operation
tool = WeaviateVectorSearchTool(
    collection_name='example_collections',
    limit=3,
    weaviate_cluster_url="https://your-weaviate-cluster-url.com",
    weaviate_api_key="your-weaviate-api-key",
)

# or 

# Setup custom model for vectorizer and generative model
tool = WeaviateVectorSearchTool(
    collection_name='example_collections',
    limit=3,
    vectorizer=Configure.Vectorizer.text2vec_openai(model="nomic-embed-text"),
    generative_model=Configure.Generative.openai(model="gpt-4o-mini"),
    weaviate_cluster_url="https://your-weaviate-cluster-url.com",
    weaviate_api_key="your-weaviate-api-key",
)

# Adding the tool to an agent
rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions with the help of the WeaviateVectorSearchTool.",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

## Arguments
- `collection_name` : The name of the collection to search within. (Required)
- `weaviate_cluster_url` : The URL of the Weaviate cluster. (Required)
- `weaviate_api_key` : The API key for the Weaviate cluster. (Required)
- `limit` : The number of results to return. (Optional)
- `vectorizer` : The vectorizer to use. (Optional)
- `generative_model` : The generative model to use. (Optional)

Preloading the Weaviate database with documents:

```python
from crewai_tools import WeaviateVectorSearchTool

# Use before hooks to generate the documents and add them to the Weaviate database. Follow the weaviate docs: https://weaviate.io/developers/wcs/connect
test_docs = client.collections.get("example_collections")


docs_to_load = os.listdir("knowledge")
with test_docs.batch.dynamic() as batch:
    for d in docs_to_load:
        with open(os.path.join("knowledge", d), "r") as f:
            content = f.read()
        batch.add_object(
            {
                "content": content,
                "year": d.split("_")[0],
            }
        )
tool = WeaviateVectorSearchTool(collection_name='example_collections', limit=3)

```
