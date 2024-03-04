# PDFSearchTool

!!! note "Depend on OpenAI"
    All RAG tools at the moment can only use openAI to generate embeddings, we are working on adding support for other providers.

!!! note "Experimental"
    We are still working on improving tools, so there might be unexpected behavior or changes in the future.

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
