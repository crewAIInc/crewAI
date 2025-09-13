# ContextualAICreateAgentTool

## Description
This tool is designed to integrate Contextual AI's enterprise-grade RAG agents with CrewAI. This tool enables you to create a new Contextual RAG agent. It uploads your documents to create a datastore and returns the Contextual agent ID and datastore ID.

## Installation
To incorporate this tool into your project, follow the installation instructions below:

```
pip install 'crewai[tools]' contextual-client
```

**Note**: You'll need a Contextual AI API key. Sign up at [app.contextual.ai](https://app.contextual.ai) to get your free API key.

## Example

```python
from crewai_tools import ContextualAICreateAgentTool

# Initialize the tool
tool = ContextualAICreateAgentTool(api_key="your_api_key_here")

# Create agent with documents
result = tool._run(
    agent_name="Financial Analysis Agent",
    agent_description="Agent for analyzing financial documents",
    datastore_name="Financial Reports",
    document_paths=["/path/to/report1.pdf", "/path/to/report2.pdf"],
)
print(result)
```

## Parameters
- `api_key`: Your Contextual AI API key
- `agent_name`: Name for the new agent
- `agent_description`: Description of the agent's purpose
- `datastore_name`: Name for the document datastore
- `document_paths`: List of file paths to upload

Example result: 

```
Successfully created agent 'Research Analyst' with ID: {created_agent_ID} and datastore ID: {created_datastore_ID}. Uploaded 5 documents.
```

You can use `ContextualAIQueryTool` with the returned IDs to query the knowledge base and retrieve relevant information from your documents.

## Key Features
- **Complete Pipeline Setup**: Creates datastore, uploads documents, and configures agent in one operation
- **Document Processing**: Leverages Contextual AI's powerful parser to ingest complex PDFs and documents
- **Vector Storage**: Use Contextual AI's datastore for large document collections

## Use Cases
- Set up new RAG agents from scratch with complete automation
- Upload and organize document collections into structured datastores
- Create specialized domain agents for legal, financial, technical, or research workflows

For more detailed information about Contextual AI's capabilities, visit the [official documentation](https://docs.contextual.ai).