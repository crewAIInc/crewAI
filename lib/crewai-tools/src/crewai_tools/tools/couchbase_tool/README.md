# CouchbaseFTSVectorSearchTool
## Description
Couchbase is a NoSQL database with vector search capabilities. Users can store and query vector embeddings. You can learn more about Couchbase vector search here: https://docs.couchbase.com/cloud/vector-search/vector-search.html 

This tool is specifically crafted for performing semantic search using Couchbase. Use this tool to find semantically similar docs to a given query.

## Installation
Install the crewai_tools package by executing the following command in your terminal:

```shell
uv pip install 'crewai[tools]'
```

## Setup
Before instantiating the tool, you need a Couchbase cluster. 
- Create a cluster on [Couchbase Capella](https://docs.couchbase.com/cloud/get-started/create-account.html), Couchbase's cloud database solution.
- Create a [local Couchbase server](https://docs.couchbase.com/server/current/getting-started/start-here.html). 

You will need to create a bucket, scope and collection on the cluster. Then, [follow this guide](https://docs.couchbase.com/python-sdk/current/hello-world/start-using-sdk.html) to create a Couchbase Cluster object and load documents into your collection.

Follow the docs below to create a vector search index on Couchbase.
- [Create a vector search index on Couchbase Capella.](https://docs.couchbase.com/cloud/vector-search/create-vector-search-index-ui.html)
- [Create a vector search index on your local Couchbase server.](https://docs.couchbase.com/server/current/vector-search/create-vector-search-index-ui.html)

Ensure that the `Dimension` field in the index matches the embedding model. For example, OpenAI's `text-embedding-3-small` model has an embedding dimension of 1536 dimensions, and so the `Dimension` field must be 1536 in the index.

## Example
To utilize the CouchbaseFTSVectorSearchTool for different use cases, follow these examples:

```python
from crewai_tools import CouchbaseFTSVectorSearchTool

# Instantiate a Couchbase Cluster object from the Couchbase SDK

tool = CouchbaseFTSVectorSearchTool(
    cluster=cluster,
    collection_name="collection",
    scope_name="scope",
    bucket_name="bucket",
    index_name="index",
    embedding_function=embed_fn
)

# Adding the tool to an agent
rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions with the help of the CouchbaseFTSVectorSearchTool.",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

## Arguments
- `cluster`: An initialized Couchbase `Cluster` instance. 
- `bucket_name`: The name of the Couchbase bucket. 
- `scope_name`: The name of the scope within the bucket. 
- `collection_name`: The name of the collection within the scope. 
- `index_name`: The name of the search index (vector index). 
- `embedding_function`: A function that takes a string and returns its embedding (list of floats). 
- `embedding_key`: Name of the field in the search index storing the vector. (Optional, defaults to 'embedding')
- `scoped_index`: Whether the index is scoped (True) or cluster-level (False). (Optional, defaults to True)
- `limit`: The maximum number of search results to return. (Optional, defaults to 3)