# ContextualAIQueryTool

## Description
This tool is designed to integrate Contextual AI's enterprise-grade RAG agents with CrewAI. Run this tool to query existing Contextual AI RAG agents that have been pre-configured with documents and knowledge bases.

## Installation
To incorporate this tool into your project, follow the installation instructions below:

```shell
pip install 'crewai[tools]' contextual-client
```

**Note**: You'll need a Contextual AI API key. Sign up at [app.contextual.ai](https://app.contextual.ai) to get your free API key.

## Example

Make sure you have already created a Contextual agent and ingested documents into the datastore before using this tool. 

```python
from crewai_tools import ContextualAIQueryTool

# Initialize the tool
tool = ContextualAIQueryTool(api_key="your_api_key_here")

# Query the agent with IDs
result = tool._run(
    query="What are the key findings in the financial report?",
    agent_id="your_agent_id_here",
    datastore_id="your_datastore_id_here"  # Optional: for document readiness checking
)
print(result)
```

The result will contain the generated answer to the user's query. 

## Parameters
**Initialization:**
- `api_key`: Your Contextual AI API key

**Query (_run method):**
- `query`: The question or query to send to the agent
- `agent_id`: ID of the existing Contextual AI agent to query (required)
- `datastore_id`: Optional datastore ID for document readiness verification (if not provided, document status checking is disabled with a warning)

## Key Features
- **Document Readiness Checking**: Automatically waits for documents to be processed before querying
- **Grounded Responses**: Built-in grounding ensures factual, source-attributed answers

## Use Cases
- Query pre-configured RAG agents with document collections
- Access enterprise knowledge bases through user queries
- Build specialized domain experts with access to curated documents

For more detailed information about Contextual AI's capabilities, visit the [official documentation](https://docs.contextual.ai).