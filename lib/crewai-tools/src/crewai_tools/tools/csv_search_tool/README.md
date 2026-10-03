# CSVSearchTool

## Description

This tool is used to perform a RAG (Retrieval-Augmented Generation) search within a CSV or Excel file's content. It allows users to semantically search for queries in the content of a specified `.csv`, `.xls`, or `.xlsx` file. This feature is particularly useful for extracting information from large tabular datasets where traditional search methods might be inefficient. All tools with "Search" in their name, including CSVSearchTool, are RAG tools designed for searching different sources of data.

## Installation

Install the crewai_tools package

```shell
pip install 'crewai[tools]'
```

## Example

```python
from crewai_tools import CSVSearchTool

# Initialize the tool with one file. The agent can only search that file.
csv_tool = CSVSearchTool(csv='path/to/your/data.csv')
xls_tool = CSVSearchTool(csv='path/to/your/workbook.xls')
xlsx_tool = CSVSearchTool(csv='path/to/your/workbook.xlsx')

# OR

# Initialize the tool without a file. The agent provides the path at runtime.
tool = CSVSearchTool()
```

## Arguments

- `csv` : The path to the `.csv`, `.xls`, or `.xlsx` file you want to search. This is a mandatory argument if the tool was initialized without a specific file; otherwise, it is optional.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = CSVSearchTool(
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
