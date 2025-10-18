# PDFSearchTool

## Description
The PDFSearchTool is a RAG tool designed for semantic searches within PDF content. It allows for inputting a search query and a PDF document, leveraging advanced search techniques to find relevant content efficiently. This capability makes it especially useful for extracting specific information from large PDF files quickly.

## Installation
To get started with the PDFSearchTool, first, ensure the crewai_tools package is installed with the following command:

```shell
pip install 'crewai[tools]'
```

## Example
Here's how to use the PDFSearchTool to search within a PDF document:

```python
from crewai_tools import PDFSearchTool

# Initialize the tool allowing for any PDF content search if the path is provided during execution
tool = PDFSearchTool()

# OR

# Initialize the tool with a specific PDF path for exclusive search within that document
tool = PDFSearchTool(pdf='path/to/your/document.pdf')
```

## Arguments
- `pdf`: **Optinal** The PDF path for the search. Can be provided at initialization or within the `run` method's arguments. If provided at initialization, the tool confines its search to the specified document.

## Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

```python
tool = PDFSearchTool(
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
