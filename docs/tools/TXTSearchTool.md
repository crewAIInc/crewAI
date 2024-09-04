# TXTSearchTool

!!! note "Experimental"
    This tool is currently in an experimental stage. We are actively working on improvements, so there might be unexpected behavior or changes in future versions.

## Description
The TXTSearchTool is designed to perform RAG (Retrieval-Augmented Generation) searches within the content of text files. It enables semantic searching of queries within specified text file content, making it an invaluable resource for quickly extracting information or locating specific sections of text based on the provided query.

## Installation
To use the TXTSearchTool, you need to install the crewai_tools package. Use pip, the Python package manager, by running the following command in your terminal or command prompt:

```shell
pip install 'crewai[tools]'
```

This command will install the TXTSearchTool along with all necessary dependencies.

## Usage

### Basic Initialization
There are two ways to initialize the TXTSearchTool:

1. Without specifying a text file:
   ```python
   from crewai_tools import TXTSearchTool

   # Initialize the tool to search within any text file's content the agent learns about during its execution
   tool = TXTSearchTool()
   ```

2. With a specific text file:
   ```python
   from crewai_tools import TXTSearchTool

   # Initialize the tool with a specific text file
   tool = TXTSearchTool(txt='path/to/text/file.txt')
   ```

### Arguments
- `txt` (str, optional): The path to the text file you want to search. If not provided during initialization, it must be specified when using the tool.

### Performing a Search
After initialization, you can use the tool to perform searches. The exact method to do this depends on how you're using the tool within your CrewAI setup.

## Advanced Configuration: Custom Models and Embeddings

By default, the TXTSearchTool uses OpenAI for both embeddings and summarization. However, you can customize these settings using a configuration dictionary:

```python
tool = TXTSearchTool(
    config=dict(
        llm=dict(
            provider="ollama",  # Options: google, openai, anthropic, llama2, etc.
            config=dict(
                model="llama2",
                # Uncomment and adjust these optional parameters as needed:
                # temperature=0.5,
                # top_p=1,
                # stream=True,
            ),
        ),
        embedder=dict(
            provider="google",  # Options: openai, ollama, etc.
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # Uncomment if needed:
                # title="Embeddings",
            ),
        ),
    )
)
```

This configuration allows you to specify different providers and models for both the language model (llm) and the embedder.

## Best Practices
- Ensure the text file is in a readable format and encoding.
- For large text files, consider splitting them into smaller, more manageable chunks.
- Experiment with different query formulations to get the most relevant results.

## Limitations
- The tool's effectiveness may vary depending on the size and complexity of the text file.
- Performance can be affected by the chosen language model and embedding provider.

## Troubleshooting
If you encounter issues:
1. Verify that the text file path is correct and accessible.
2. Check that you have the necessary permissions to read the file.
3. Ensure you have a stable internet connection if using cloud-based models.

For further assistance, please refer to the CrewAI documentation or reach out to the support community.
