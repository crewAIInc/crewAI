# JSONSearchTool

!!! note "Experimental Status"
    The JSONSearchTool is currently in an experimental phase. This means the tool is under active development, and users might encounter unexpected behavior or changes. We highly encourage feedback on any issues or suggestions for improvements.

## Description
The JSONSearchTool is designed to facilitate efficient and precise searches within JSON file contents. It utilizes a RAG (Retrieve and Generate) search mechanism, allowing users to specify a JSON path for targeted searches within a particular JSON file. This capability significantly improves the accuracy and relevance of search results.

## Installation
To install the JSONSearchTool, use the following pip command:

```shell
pip install 'crewai[tools]'
```

## Usage Examples
Here are updated examples on how to utilize the JSONSearchTool effectively for searching within JSON files. These examples take into account the current implementation and usage patterns identified in the codebase.

```python
from crewai.json_tools import JSONSearchTool  # Updated import path

# General JSON content search
# This approach is suitable when the JSON path is either known beforehand or can be dynamically identified.
tool = JSONSearchTool()

# Restricting search to a specific JSON file
# Use this initialization method when you want to limit the search scope to a specific JSON file.
tool = JSONSearchTool(json_path='./path/to/your/file.json')
```

## Arguments
- `json_path` (str, optional): Specifies the path to the JSON file to be searched. This argument is not required if the tool is initialized for a general search. When provided, it confines the search to the specified JSON file.

## Configuration Options
The JSONSearchTool supports extensive customization through a configuration dictionary. This allows users to select different models for embeddings and summarization based on their requirements.

```python
tool = JSONSearchTool(
    config={
        "llm": {
            "provider": "ollama",  # Other options include google, openai, anthropic, llama2, etc.
            "config": {
                "model": "llama2",
                # Additional optional configurations can be specified here.
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            },
        },
        "embedder": {
            "provider": "google",
            "config": {
                "model": "models/embedding-001",
                "task_type": "retrieval_document",
                # Further customization options can be added here.
            },
        },
    }
)
```