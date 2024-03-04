# TXTSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is used to perform a RAG (Retrieval-Augmented Generation) search within the content of a text file. It allows for semantic searching of a query within a specified text file's content, making it an invaluable resource for quickly extracting information or finding specific sections of text based on the query provided.

## Installation
To use the TXTSearchTool, you first need to install the crewai_tools package. This can be done using pip, a package manager for Python. Open your terminal or command prompt and enter the following command:

```shell
pip install 'crewai[tools]'
```

This command will download and install the TXTSearchTool along with any necessary dependencies.

## Example
The following example demonstrates how to use the TXTSearchTool to search within a text file. This example shows both the initialization of the tool with a specific text file and the subsequent search within that file's content.

```python
from crewai_tools import TXTSearchTool

# Initialize the tool to search within any text file's content the agent learns about during its execution
tool = TXTSearchTool()

# OR

# Initialize the tool with a specific text file, so the agent can search within the given text file's content
tool = TXTSearchTool(txt='path/to/text/file.txt')
```

## Arguments
- `txt` (str): **Optinal**. The path to the text file you want to search. This argument is only required if the tool was not initialized with a specific text file; otherwise, the search will be conducted within the initially provided text file.
