# JSONSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## Description
This tool is used to perform a RAG search within a JSON file's content. It allows users to initiate a search with a specific JSON path, focusing the search operation within that particular JSON file. If the path is provided at initialization, the tool restricts its search scope to the specified JSON file, thereby enhancing the precision of search results.

## Installation
Install the crewai_tools package by executing the following command in your terminal:

```shell
pip install 'crewai[tools]'
```

## Example
Below are examples demonstrating how to use the JSONSearchTool for searching within JSON files. You can either search any JSON content or restrict the search to a specific JSON file.

```python
from crewai_tools import JSONSearchTool

# Example 1: Initialize the tool for a general search across any JSON content. This is useful when the path is known or can be discovered during execution.
tool = JSONSearchTool()

# Example 2: Initialize the tool with a specific JSON path, limiting the search to a particular JSON file.
tool = JSONSearchTool(json_path='./path/to/your/file.json')
```

## Arguments
- `json_path` (str): An optional argument that defines the path to the JSON file to be searched. This parameter is only necessary if the tool is initialized without a specific JSON path. Providing this argument restricts the search to the specified JSON file.