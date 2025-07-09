# MongoDBVectorSearchTool

## Description
This tool is specifically crafted for conducting vector searches within docs within a MongoDB database. Use this tool to find semantically similar docs to a given query.

MongoDB can act as a vector database that is used to store and query vector embeddings. You can follow the docs here:
https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/

## Installation
Install the crewai_tools package with MongoDB support by executing the following command in your terminal:

```shell
pip install crewai-tools[mongodb]
```

or

```
uv add crewai-tools --extra mongodb
```

## Example
To utilize the MongoDBVectorSearchTool for different use cases, follow these examples:

```python
from crewai_tools import MongoDBVectorSearchTool

# To enable the tool to search any website the agent comes across or learns about during its operation
tool = MongoDBVectorSearchTool(
    database_name="example_database',
    collection_name='example_collections',
    connection_string="<your_mongodb_connection_string>",
)
```

or

```python
from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool

# Setup custom embedding model and customize the parameters.
query_config = MongoDBVectorSearchConfig(limit=10, oversampling_factor=2)
tool = MongoDBVectorSearchTool(
    database_name="example_database',
    collection_name='example_collections',
    connection_string="<your_mongodb_connection_string>",
    query_config=query_config,
    index_name="my_vector_index",
    generative_model="gpt-4o-mini"
)

# Adding the tool to an agent
rag_agent = Agent(
    name="rag_agent",
    role="You are a helpful assistant that can answer questions with the help of the MongoDBVectorSearchTool.",
    goal="...",
    backstory="...",
    llm="gpt-4o-mini",
    tools=[tool],
)
```

Preloading the MongoDB database with documents:

```python
from crewai_tools import MongoDBVectorSearchTool

# Generate the documents and add them to the MongoDB database
test_docs = client.collections.get("example_collections")

# Create the tool.
tool = MongoDBVectorSearchTool(
    database_name="example_database',
    collection_name='example_collections',
    connection_string="<your_mongodb_connection_string>",
)

# Add the text from a set of CrewAI knowledge documents.
texts = []
for d in os.listdir("knowledge"):
    with open(os.path.join("knowledge", d), "r") as f:
        texts.append(f.read())
tool.add_texts(text)

# Create the vector search index (if it wasn't already created in Atlas).
tool.create_vector_search_index(dimensions=3072)
```
