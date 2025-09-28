# DirectorySearchTool

## Description
This tool is designed to perform a semantic search for queries within the content of a specified directory. Utilizing the RAG (Retrieval-Augmented Generation) methodology, it offers a powerful means to semantically navigate through the files of a given directory. The tool can be dynamically set to search any directory specified at runtime or can be pre-configured to search within a specific directory upon initialization.

## Installation
To start using the DirectorySearchTool, you need to install the crewai_tools package. Execute the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
The following examples demonstrate how to initialize the DirectorySearchTool for different use cases and how to perform a search:

```python
from crewai_tools import DirectorySearchTool

# To enable searching within any specified directory at runtime
tool = DirectorySearchTool()

# Alternatively, to restrict searches to a specific directory
tool = DirectorySearchTool(directory='/path/to/directory')
```

## Arguments
- `directory` : This string argument specifies the directory within which to search. It is mandatory if the tool has not been initialized with a directory; otherwise, the tool will only search within the initialized directory.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = DirectorySearchTool(
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
