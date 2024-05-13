# DirectorySearchTool

!!! note "Experimental"
    The DirectorySearchTool is under continuous development. Features and functionalities might evolve, and unexpected behavior may occur as we refine the tool.

## Description
The DirectorySearchTool enables semantic search within the content of specified directories, leveraging the Retrieval-Augmented Generation (RAG) methodology for efficient navigation through files. Designed for flexibility, it allows users to dynamically specify search directories at runtime or set a fixed directory during initial setup.

## Installation
To use the DirectorySearchTool, begin by installing the crewai_tools package. Execute the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Initialization and Usage
Import the DirectorySearchTool from the `crewai_tools` package to start. You can initialize the tool without specifying a directory, enabling the setting of the search directory at runtime. Alternatively, the tool can be initialized with a predefined directory.

```python
from crewai_tools import DirectorySearchTool

# For dynamic directory specification at runtime
tool = DirectorySearchTool()

# For fixed directory searches
tool = DirectorySearchTool(directory='/path/to/directory')
```

## Arguments
- `directory`: A string argument that specifies the search directory. This is optional during initialization but required for searches if not set initially.

## Custom Model and Embeddings
The DirectorySearchTool uses OpenAI for embeddings and summarization by default. Customization options for these settings include changing the model provider and configuration, enhancing flexibility for advanced users.

```python
tool = DirectorySearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # Options include ollama, google, anthropic, llama2, and more
            config=dict(
                model="llama2",
                # Additional configurations here
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```