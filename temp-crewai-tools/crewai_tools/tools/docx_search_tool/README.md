# DOCXSearchTool

## Description
The DOCXSearchTool is a RAG tool designed for semantic searching within DOCX documents. It enables users to effectively search and extract relevant information from DOCX files using query-based searches. This tool is invaluable for data analysis, information management, and research tasks, streamlining the process of finding specific information within large document collections.

## Installation
Install the crewai_tools package by running the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
The following example demonstrates initializing the DOCXSearchTool to search within any DOCX file's content or with a specific DOCX file path.

```python
from crewai_tools import DOCXSearchTool

# Initialize the tool to search within any DOCX file's content
tool = DOCXSearchTool()

# OR

# Initialize the tool with a specific DOCX file, so the agent can only search the content of the specified DOCX file
tool = DOCXSearchTool(docx='path/to/your/document.docx')
```

## Arguments
- `docx`: An optional file path to a specific DOCX document you wish to search. If not provided during initialization, the tool allows for later specification of any DOCX file's content path for searching.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = DOCXSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```
