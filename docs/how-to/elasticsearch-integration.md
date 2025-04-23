# Elasticsearch Integration

CrewAI supports using Elasticsearch as an alternative to ChromaDB for RAG (Retrieval Augmented Generation) storage. This allows you to leverage Elasticsearch's powerful search capabilities and scalability for your AI agents.

## Installation

To use Elasticsearch with CrewAI, you need to install the Elasticsearch Python client:

```bash
pip install elasticsearch
```

## Using Elasticsearch for Memory

You can configure your crew to use Elasticsearch for memory storage:

```python
from crewai import Agent, Crew, Task

# Create agents and tasks
agent = Agent(
    role="Researcher",
    goal="Research a topic",
    backstory="You are a researcher who loves to find information.",
)

task = Task(
    description="Research about AI",
    expected_output="Information about AI",
    agent=agent,
)

# Create a crew with Elasticsearch memory
crew = Crew(
    agents=[agent],
    tasks=[task],
    memory_config={
        "provider": "elasticsearch",
        "host": "localhost",  # Optional, defaults to localhost
        "port": 9200,         # Optional, defaults to 9200
        "username": "user",   # Optional
        "password": "pass",   # Optional
    },
)

# Execute the crew
result = crew.kickoff()
```

## Using Elasticsearch for Knowledge

You can also use Elasticsearch for knowledge storage:

```python
from crewai import Agent, Crew, Task
from crewai.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# Create knowledge with Elasticsearch storage
content = "AI is a field of computer science that focuses on creating machines that can perform tasks that typically require human intelligence."
string_source = StringKnowledgeSource(
    content=content, metadata={"topic": "AI"}
)

knowledge = Knowledge(
    collection_name="test",
    sources=[string_source],
    storage_provider="elasticsearch",  # Use Elasticsearch
    # Optional Elasticsearch configuration
    host="localhost",
    port=9200,
    username="user",
    password="pass",
)

# Create an agent with the knowledge
agent = Agent(
    role="AI Expert",
    goal="Explain AI",
    backstory="You are an AI expert who loves to explain AI concepts.",
    knowledge=[knowledge],
)

# Create a task
task = Task(
    description="Explain what AI is",
    expected_output="Explanation of AI",
    agent=agent,
)

# Create a crew
crew = Crew(
    agents=[agent],
    tasks=[task],
)

# Execute the crew
result = crew.kickoff()
```

## Configuration Options

The Elasticsearch integration supports the following configuration options:

- `host`: Elasticsearch host (default: "localhost")
- `port`: Elasticsearch port (default: 9200)
- `username`: Elasticsearch username (optional)
- `password`: Elasticsearch password (optional)
- Additional keyword arguments are passed directly to the Elasticsearch client

## Running Tests

To run the Elasticsearch tests, you need to set the `RUN_ELASTICSEARCH_TESTS` environment variable to `true`:

```bash
RUN_ELASTICSEARCH_TESTS=true pytest tests/memory/elasticsearch_storage_test.py tests/knowledge/elasticsearch_knowledge_storage_test.py tests/integration/elasticsearch_integration_test.py
```
